using Distributed
using LatinHypercubeSampling
using Plots

if length(workers()) == 1
    addprocs(Sys.CPU_THREADS)
    # addprocs(8)
end
# include("PHeuristics.jl")

@everywhere begin
    include("PHeuristics.jl")
    ρ = 0.8
    game_size = 3
    n_games = 1000
    level = 3
    # n_inits = 8
    n_inits = Sys.CPU_THREADS
    bounds = Bounds([0., -1., 0.], [1., 1., 10.])
    costs = Costs(0.1, 0.05)
    opp_h = Heuristic(0.5, 3., 2.)
    opp_h = Heuristic(0., 0., 5.)
    # opp_h = Heuristic(0., -1, 2.)
end

# hline!(s)

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

function sample_init(n, level)
    n -= 1 # because we add 0.1s later
    X = (LHCoptim(n, 3*level, 1000)[1] .- 1) ./ n
    init = [bounds(X[i, :]) .+ 0.001 for i in 1:size(X)[1]]
    push!(init, 0.001 * ones(3*level))
end

sample_init(5, 1)
@time res = pmap(sample_init(n_inits, level)) do x
    h = optimize_h(level, train_games, train_opp_plays, costs; init_x=x)
    train_score = -loss(h, train_games, train_opp_plays, costs)
    test_score = -loss(h, test_games, test_opp_plays, costs)
    # println("Train score: ", train_score, "   Test score:  ", test_score)
    (h, train_score, test_score)
end

prisoners_dilemma = Game([[3 0];[4 1]], [[3 4]; [0 1]])
prisoners_dilemma = Game(prisoners_dilemma.row .*2, prisoners_dilemma.col .*2)

centipede_game = Game(
    [[2 2 2 2 2]; [1 4 4 4 4]; [1 3 10 10 10]; [1 3 5 18 18]; [1 3 5 7 30]],
    [[0 0 0 0 0]; [3 1 1 1 1]; [3 7 4 4 4]; [3 7 13 6 6]; [3 7 13 23 8]]
)

sort!(res, by= x -> x[2], rev=true)

for (h, trn, tst) in res
    println("----- ", h, " -----")
    println(@sprintf "Train: %.3f   Test: %.3f" trn tst)
    println("PD behavior: ", decide_probs(h, prisoners_dilemma))
end

println("Best h on training", res[1])
sort!(res, by= x -> x[3], rev=true)
println("Best h on test", res[1])
@printf("Level-2 exact perf: %.3f", loss(opt_level_2, train_games, train_opp_plays, costs))
@printf("Level-2 cheap perf: %.3f", loss(opt_level_2_cheap, train_games, train_opp_plays, costs))


#%%
@printf("Level-2 exact perf: %.3f \n", loss(opt_level_2, train_games, train_opp_plays, costs))
@printf("Level-2 cheap perf: %.3f \n", loss(opt_level_2_cheap, train_games, train_opp_plays, costs))


best_1 = optimize_h(1, train_games, train_opp_plays, costs)
@printf("Best level_1 cheap perf: %.3f , %s \n", loss(best_1, train_games, train_opp_plays, costs), best_1)



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
