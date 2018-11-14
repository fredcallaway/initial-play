using Distributed

@everywhere include("PHeuristics.jl")


h = Heuristic(0, 0, 0.0)
g = Game(3, 0.)
g.row[1,2] = 100
g
relative_values(h, g)

s = SimHeuristic(h_dists, 2)
s3 = SimHeuristic(h_dists, 3)

prisoners_dilemma = Game([[3 0];[4 1]], [[3 4]; [0 1]])
prisoners_dilemma = Game(prisoners_dilemma.row .*2, prisoners_dilemma.col .*2)


level_0 = Heuristic(0.5, 1., 0.0)
level_1 = Heuristic(0., 0., 10.)
noisy = Heuristic(0.2, 1., 0.9)
maximin = Heuristic(0., -10., 10.)
maxjoint = Heuristic(0.5, 10., 10.)


ρ = 0.6
n_games = 1000
game_size = 3
method = :de_rand_2_bin
opp_h = noisy
correct_level_2 = SimHeuristic([opp_h, Heuristic(0., 0., 10.)], 2)
correct_level_3 = SimHeuristic([Heuristic(0., 0., 0.), opp_h, Heuristic(0., 0., 10.)], 3)
println("Compare with ρ = ", ρ)
training_games = [Game(game_size, ρ) for i in range(1,n_games)];


test_games = [Game(game_size, ρ) for i in range(1,n_games)];
println("Level-0: \t", payoff(level_0, opp_h, training_games))
println("Level-1: \t", payoff(level_1, opp_h, training_games))
println("Level-2: \t", payoff(correct_level_2, opp_h, training_games))
println("Level-3: \t", payoff(correct_level_3, opp_h, training_games))
println("Maximin: \t", payoff(maximin, opp_h, training_games))
println("Max joint: \t",payoff(maxjoint, opp_h, training_games))

opt_h!(h, opp_h, training_games, h_dists; max_time=10., method=method)
println(h, "    ", payoff(h, opp_h, training_games))
opt_s!(s, opp_h, training_games, h_dists; max_time=20., method=method)
println(s, "    ", payoff(s, opp_h, training_games))
opt_s!(s3, opp_h, training_games, h_dists; max_time=30., method=method)
println(s3,"    ",  payoff(s3, opp_h, training_games))


println("Test games:")

println("Best h: \t", payoff(h, opp_h, test_games))
println("Best s2: \t", payoff(s, opp_h, test_games))
println("Best s3: \t", payoff(s3, opp_h, test_games))
println("Level-0: \t", payoff(level_0, opp_h, test_games))
println("Level-1: \t", payoff(level_1, opp_h, test_games))
# println("Level-2: \t", payoff(level_2, opp_h, test_games))
println("Maximin: \t", payoff(maximin, opp_h, test_games))
println("Sum joint: \t",payoff(maxjoint, opp_h, test_games))


println(@sprintf("PD for h: %0.f", decide(h, prisoners_dilemma)))
println(@sprintf("PD for s2: %0.f", decide(s, prisoners_dilemma)))
println(@sprintf("PD for s3: %0.f", decide(s3, prisoners_dilemma)))

for ρ in -1.:0.2:1.
    println("----- ρ=", ρ, " -----")
    h = Heuristic(h_dists)
    games = [Game(game_size, ρ) for i in range(1,n_games)]
    opt_h!(h, opp_h, games, h_dists; max_time=10., method=method)
    println(h, "    ", payoff(h, opp_h, games))
    println(@sprintf("PD for h: %0.f", decide(h, prisoners_dilemma)))
    println("Level-1: \t", payoff(level_1, opp_h, games))
end

ρ = -0.5
games = [Game(game_size, ρ) for i in range(1,n_games)]
for method in [:separable_nes, :de_rand_1_bin, :de_rand_2_bin, :generating_set_search, :probabilistic_descent, :resampling_memetic_search, :generating_set_search]
    h = Heuristic(h_dists)
    opt_h!(h, opp_h, games, h_dists; max_time=10., method=method, trace=:silent)
    println("Using optimizer: ", method)
    println(h, "    ", payoff(h, opp_h, games))
end
