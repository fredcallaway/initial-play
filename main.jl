
import Base: rand
using Distributed
if length(workers()) == 1
    addprocs(Sys.CPU_THREADS)
    # addprocs(8)
end
@everywhere include("PHeuristics.jl")
# include("PHeuristics.jl")

@everywhere begin
    ρ = 0.
    game_size = 3
    n_games = 1000
    level = 2
    # n_inits = 8
    n_inits = Sys.CPU_THREADS
    bounds = Bounds([0., -1., 0.], [1., 1., 10.])
    costs = Costs(0.1, 0.05)
    opp_h = Heuristic(0, 0, 10)
end

train_games = [Game(game_size, ρ) for i in range(1,n_games)]
train_opp_plays = opp_cols_played(opp_h, train_games)
test_games = [Game(game_size, ρ) for i in range(1,n_games)]
test_opp_plays = opp_cols_played(opp_h, test_games);

opt_level_2 = SimHeuristic([opp_h, Heuristic(0.,0.,4.)])
opt_level_2_cheap = SimHeuristic([Heuristic(0.,0.,2.), Heuristic(0.,0.,3.)])

@everywhere begin
    train_games = $train_games
    test_games = $test_games
    train_opp_plays = $train_opp_plays
    test_opp_plays = $test_opp_plays

end

x_inits = [rand(bounds, level) for i in 1:n_inits]
@time res = pmap(x_inits) do x
    h = optimize_h(level, train_games, train_opp_plays, costs; init_x=x)
    train_score = -loss(h, train_games, train_opp_plays, costs)
    test_score = -loss(h, test_games, test_opp_plays, costs)
    # println("Train score: ", train_score, "   Test score:  ", test_score)
    (h, train_score, test_score)
end

prisoners_dilemma = Game([[3 0];[4 1]], [[3 4]; [0 1]])
prisoners_dilemma = Game(prisoners_dilemma.row .*2, prisoners_dilemma.col .*2)

for (h, trn, tst) in res
    println("----- ", h, " -----")
    println(@sprintf "Train: %.3f   Test: %.3f" trn tst)
    println("PD behavior: ", decide_probs(h, prisoners_dilemma))
end

sort!(res, by= x -> x[2], rev=true)
println("Best h on training", res[1])
sort!(res, by= x -> x[3], rev=true)
println("Best h on test", res[1])
@printf("Level-2 exact perf: %.0f", loss(level_2, train_games, train_opp_plays, costs))
@printf("Level-2 cheap perf: %.0f", loss(level_2_cheap, train_games, train_opp_plays, costs))


optimize_h(level, train_games, opp_plays, costs)


#
# ρ = 0.5
# game_size = 3
# n_games = 1000
# level = 2
# games = [Game(game_size, ρ) for i in range(1,n_games)]
# opp_h = Heuristic(0, 0, 10)
# opp_plays = opp_cols_played(opp_h, games)
# costs = Costs(0.01, 0.01)
# x_inits = [rand(3 * level) for i in 1:3]
# opt_hs = map(x -> optimize_h(level, games, opp_plays, costs; init_x=x), x_inits)
# # rand(3) isa Vector{Real}
