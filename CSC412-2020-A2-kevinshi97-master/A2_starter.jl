using Revise # lets you change A2funcs without restarting julia!
includet("A2_src.jl")
using Plots
using Statistics: mean
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!

function log_prior(zs)
  return factorized_gaussian_log_density(0, 0, zs)
end

function logp_a_beats_b(za,zb)
  return -log1pexp(-(za-zb))
end

function all_games_log_likelihood(zs,games)
  zs_a = zs[games[:,1], :]
  zs_b =  zs[games[:,2], :]
  likelihoods =  logp_a_beats_b.(zs_a,zs_b)
  return  sum(likelihoods, dims = 1)
end

function joint_log_density(zs,games)
  return log_prior(zs) + all_games_log_likelihood(zs, games)
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)
# Example for how to use contour plotting code
plot(title = "Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
# skillcontour!(example_gaussian; label="example gaussian")
skillcontour!(example_gaussian)
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.pdf"))

# TODO: plot prior contours
plot(title = "Joint Prior Contour Plot",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint_prior(zs) = exp(log_prior(zs))
skillcontour!(joint_prior)
plot_line_equal_skill!()
savefig(joinpath("plots","2a_prior_contours.pdf"))

# TODO: plot likelihood contours
plot(title = "Likelihood that A Beats B",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
likelihood_a_beats_b(zs) = exp.(logp_a_beats_b.(zs[1,:],zs[2,:]))
skillcontour!(likelihood_a_beats_b)
plot_line_equal_skill!()
savefig(joinpath("plots","2b_likelihood_contours.pdf"))

# TODO: plot joint contours with player A winning 1 game
plot(title = "Joint Posterior Given That A Beat B",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint_posterior(zs) = exp(joint_log_density(zs, two_player_toy_games(1,0)))
skillcontour!(joint_posterior)
plot_line_equal_skill!()
savefig(joinpath("plots","2c_A_beat_B_once_contours.pdf"))

# TODO: plot joint contours with player A winning 10 games
plot(title = "Joint Posterior Given That A Beat B 10 Times",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint_posterior(zs) = exp(joint_log_density(zs, two_player_toy_games(10,0)))
skillcontour!(joint_posterior)
plot_line_equal_skill!()
savefig(joinpath("plots","2d_A_beat_B_ten_times_contours.pdf"))

#TODO: plot joint contours with player A winning 10 games and player B winning 10 games
plot(title = "Joint Posterior Given That A and B Both Win 10 Times",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint_posterior(zs) = exp(joint_log_density(zs, two_player_toy_games(10,10)))
skillcontour!(joint_posterior)
plot_line_equal_skill!()
savefig(joinpath("plots","2e_A_and_B_win_ten_times_contours.pdf"))

function elbo(params,logp,num_samples)
  # μ = params[1], log(σ) = params[2]
  num_players = size(params[1])[1]
  samples = exp.(params[2]) .* randn(num_players,num_samples) .+ params[1]
  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(params[1], params[2], samples)
  return sum(logp_estimate - logq_estimate)/num_samples #should return scalar (hint: average over batch)
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  # TODO: Write a function that takes parameters for q,
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end

# Toy game
num_players_toy = 2
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)

function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10, title = "Fit Toy Variational Dist.")
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params->neg_toy_elbo(params; games = toy_evidence, num_samples = num_q_samples),params_cur)[1]
    params_cur = params_cur .- grad_params .* lr
    @info neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples)

    # This is commented out for the report so that only the final image will show
    # plot(title = title,
    #     xlabel = "Player A Skill",
    #     ylabel = "Player B Skill"
    #    )
    # joint_posterior(zs) = exp(joint_log_density(zs, toy_evidence))
    # skillcontour!(joint_posterior;colour=:red)
    # plot_line_equal_skill!()
    # iter_gaussian(zs) = exp(factorized_gaussian_log_density(params_cur[1],params_cur[2],zs))
    # display(skillcontour!(iter_gaussian;colour=:blue))  # run this line to see the model train
  end

  plot(title = title,
      xlabel = "Player A Skill",
      ylabel = "Player B Skill"
     )
  joint_posterior(zs) = exp(joint_log_density(zs, toy_evidence))
  skillcontour!(joint_posterior;colour=:red)
  plot_line_equal_skill!()
  iter_gaussian(zs) = exp(factorized_gaussian_log_density(params_cur[1],params_cur[2],zs))
  display(skillcontour!(iter_gaussian;colour=:blue))  # run this line to see the model train
  println("Final Loss: ", neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples))
  return params_cur
end

#TODO: fit q with SVI observing player A winning 1 game
#TODO: save final posterior plots
toy_games_1_0 = two_player_toy_games(1,0)
final_params = fit_toy_variational_dist(toy_params_init, toy_games_1_0; title = "Fit Toy Variational Dist. given A Beat B Once")
savefig(joinpath("plots","3d_fit_toy_A_beat_B_once.pdf"))

#TODO: fit q with SVI observing player A winning 10 games
#TODO: save final posterior plots
toy_games_10_0 = two_player_toy_games(10,0)
final_params = fit_toy_variational_dist(toy_params_init, toy_games_10_0; title = "Fit Toy Variational Dist. given A Beat B 10 Times")
savefig(joinpath("plots","3e_fit_toy_A_beat_B_ten_times.pdf"))


#TODO: fit q with SVI observing player A winning 10 games and player B winning 10 games
#TODO: save final posterior plots
toy_games_10_10 = two_player_toy_games(10,10)
final_params = fit_toy_variational_dist(toy_params_init, toy_games_10_10; title = "Fit Toy Variational Dist. given A and B Both Win 10 Times")
savefig(joinpath("plots","3f_fit_toy_A_and_B_win_ten_times.pdf"))

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")

function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params->neg_toy_elbo(params; games = tennis_games, num_samples = num_q_samples),params_cur)[1]
    params_cur = params_cur .- grad_params .* lr
    @info neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples)
  end
  println("Final Loss:", neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples))
  return params_cur
end

# TODO: Initialize variational family
# init_mu = vec(randn(num_players, 1))
# init_log_sigma = vec(randn(num_players, 1))
init_mu = vec(zeros(num_players, 1))
init_log_sigma = vec(ones(num_players, 1))
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)

perm = sortperm(trained_params[1]);
display(plot(trained_params[1][perm], yerror=exp.(trained_params[2][perm]),
    title = "Approximate Mean and Variance of All Players",
    xlabel = "Player Number",
    ylabel = "Mean Player Skill"))
savefig(joinpath("plots","4c_approx_mean_var_all_players.pdf"))

#TODO: 10 players with highest mean skill under variational model
#hint: use sortperm
top_ten_player_indices = reverse(perm)[1:10]
top_ten_player_names = player_names[top_ten_player_indices]


#TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names

i_roger_federer = findall(x -> x == "Roger-Federer",player_names)[1][1]
i_rafael_nadal  = findall(x -> x == "Rafael-Nadal", player_names)[1][1]

rf_rn_mu  = [trained_params[1][i_roger_federer]; trained_params[1][i_rafael_nadal]]
rf_rn_logsig = [trained_params[2][i_roger_federer]; trained_params[2][i_rafael_nadal]]

plot(title = "Joint Posterior of Skill Between Roger Federer and Rafael Nadal",
    xlabel = "Roger Federer's Skill",
    ylabel = "Rafael Nadal's Skill",
    legend=:topleft)
rf_rn_joint_posterior(zs) = exp(factorized_gaussian_log_density(rf_rn_mu, rf_rn_logsig, zs))
skillcontour!(rf_rn_joint_posterior)
display(plot_line_equal_skill!())
savefig(joinpath("plots","4e_joint_posterior_rf_rn.pdf"))

using Random, Distributions
Random.seed!(123)

#4g
rf_rn_mu  = [trained_params[1][i_roger_federer]; trained_params[1][i_rafael_nadal]]
rf_rn_logsig = [trained_params[2][i_roger_federer]; trained_params[2][i_rafael_nadal]]

y_mu = rf_rn_mu[1] - rf_rn_mu[2]
y_sig = sqrt(exp(rf_rn_logsig[1])^2 + exp(rf_rn_logsig[2])^2)
p_fr_gt_rn = 1 - cdf(Normal(y_mu, y_sig), 0)

num_samples = 10000
fd_rn_samples = exp.(rf_rn_logsig) .* randn(2, num_samples) .+ rf_rn_mu
mc_fd_gt_rn = sum(fd_rn_samples[1, :] .> fd_rn_samples[2, :])/num_samples

println("Exact probability under approximate posterior:", p_fr_gt_rn)
println("Simple Monte Carlo with 10000 samples under approximate posterior:", mc_fd_gt_rn)

#4h
rf_worst_mu  = [trained_params[1][i_roger_federer]; trained_params[1][perm[1]]]
rf_worst_logsig = [trained_params[2][i_roger_federer]; trained_params[2][perm[1]]]

y_mu = rf_worst_mu[1] - rf_worst_mu[2]
y_sig = sqrt(exp(rf_worst_logsig[1])^2 + exp(rf_worst_logsig[2])^2)
p_fr_gt_worst = 1 - cdf(Normal(y_mu, y_sig), 0)

num_samples = 10000
fd_worst_samples = exp.(rf_worst_logsig) .* randn(2, num_samples) .+ rf_worst_mu
mc_fd_gt_worst = sum(fd_worst_samples[1, :] .> fd_worst_samples[2, :])/num_samples

println("Exact probability under approximate posterior:", p_fr_gt_worst)
println("Simple Monte Carlo with 10000 samples under approximate posterior:", mc_fd_gt_worst)
