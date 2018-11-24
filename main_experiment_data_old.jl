using Distributed
using LatinHypercubeSampling
using Plots
import StatsBase: sample
using JSON

if length(workers()) == 1
    addprocs(Sys.CPU_THREADS)
    # addprocs(8)
end
# include("PHeuristics.jl")

# Load list with experiment names:
data_name_list = open("data/dataset_list.json") do f
        JSON.parse(read(f, String))
end

game_names = open("data/game_names_list.json") do f
    JSON.parse(read(f, String))
end

data_sets = unique(data_name_list)


# %%


@everywhere begin
    include("PHeuristics.jl")
    ρ = 0.8
    game_size = 3
    n_games = 1000
    level = 1
    # n_inits = 8
    n_inits = Sys.CPU_THREADS
    bounds = Bounds([0., -3., 0.], [1., 3., 10.])
    costs = Costs(0.1, 0.2)
    opp_h = Heuristic(0.5, 3., 2.)
    opp_h = Heuristic(0., 0., 5.)
    # opp_h = Heuristic(0., -1, 2.)
end

# Loading data
exp_games = games_from_json("data/games.json")
exp_row_plays = plays_vec_from_json("data/rows_played.json")
exp_col_plays = plays_vec_from_json("data/cols_played.json")


# %% Setting idx so train and set are the same for all optimizations

n_train = 100
train_idxs = sample(1:length(exp_games), n_train; replace=false)
test_idxs = setdiff(1:length(exp_games), train_idxs)
sort!(train_idxs)
sort!(test_idxs)

# %% Optimal play against actual data

@everywhere begin
    exp_games = $exp_games
    exp_col_plays = $exp_col_plays
    exp_row_plays = $exp_row_plays
    train_idxs = $train_idxs
    test_idxs = $test_idxs
    train_games = $exp_games[train_idxs]
    train_opp_plays = $exp_col_plays[train_idxs]
    test_games = $exp_games[test_idxs]
    test_opp_plays = $exp_col_plays[test_idxs]
    costs = Costs(0.02, 0.13)
    level = 1
end

@time opt_res1 = pmap(sample_init(n_inits, level)) do x
    try
        h = optimize_h(level, train_games, train_opp_plays, costs; init_x=x, loss_f = loss_from_dist)
        train_score = -loss_from_dist(h, train_games, train_opp_plays, costs)
        test_score = -loss_from_dist(h, test_games, test_opp_plays, costs)
        (h, train_score, test_score)
    catch err
        println(err)
    end
end


opt_res1 = filter(x -> x !== nothing, opt_res1)
sort!(opt_res1, by= x -> x[2], rev=true)


for (h, trn, tst) in opt_res1
    println("----- ", h, " -----")
    println(@sprintf "Train: %.3f   Test: %.3f" trn tst)
end
#%% Opt for level 2

@everywhere begin
    level = 2
    costs = Costs(0.015, 0.1)
end

@time opt_res2 = pmap(sample_init(n_inits, level)) do x
    try
        h = optimize_h(level, train_games, train_opp_plays, costs; init_x=x, loss_f = loss_from_dist)
        train_score = -loss_from_dist(h, train_games, train_opp_plays, costs)
        test_score = -loss_from_dist(h, test_games, test_opp_plays, costs)
        (h, train_score, test_score)
    catch err
        println(err)
    end
end

opt_res2 = filter(x -> x !== nothing, opt_res2)
sort!(opt_res2, by= x -> x[2], rev=true)


for (h, trn, tst) in opt_res2
    println("----- ", h, " -----")
    println(@sprintf "Train: %.3f   Test: %.3f" trn tst)
end


# %% Predict using MSD
@everywhere begin
    train_idxs = $train_idxs
    test_idxs = $test_idxs
    train_games = $exp_games[train_idxs]
    train_row_plays = $exp_row_plays[train_idxs]
    test_games = $exp_games[test_idxs]
    test_row_plays = $exp_row_plays[test_idxs]
    costs = Costs(0.05, 0.2)
    level = 1
end


@time msd_res = pmap(sample_init(n_inits, level)) do x
    try
        h = optimize_h(level, train_games, train_row_plays, costs; init_x=x, loss_f = pred_loss)
        train_score = pred_loss(h, train_games, train_row_plays)
        test_score = pred_loss(h, test_games, test_row_plays)
        (h, train_score, test_score)
    catch err
        println(err)
    end
end

msd_res = filter(x -> x !== nothing, msd_res)
sort!(msd_res, by= x -> x[2], rev=false)

for (h, trn, tst) in msd_res
    println("----- ", h, " -----")
    println(@sprintf "Train: %.3f   Test: %.3f" trn tst)
end


#%% Predicting using maximum likelihood
@everywhere begin
    train_idxs = $train_idxs
    test_idxs = $test_idxs
    train_games = $exp_games[train_idxs]
    train_row_plays = $exp_row_plays[train_idxs]
    test_games = $exp_games[test_idxs]
    test_row_plays = $exp_row_plays[test_idxs]
    costs = Costs(0.05, 0.2)
    level = 1
end


@time ML_res = pmap(sample_init(n_inits, level)) do x
    try
        h = optimize_h(level, train_games, train_row_plays, costs; init_x=x, loss_f = pred_likelihood)
        train_score = -pred_likelihood(h, train_games, train_row_plays, costs)
        test_score = -pred_likelihood(h, test_games, test_row_plays, costs)
        (h, train_score, test_score)
    catch err
        println(err)
    end
end

ML_res = filter(x -> x !== nothing, ML_res)
sort!(ML_res, by= x -> x[2], rev=true)

for (h, trn, tst) in ML_res
    println("----- ", h, " -----")
    println(@sprintf "Train: %.3f   Test: %.3f" trn tst)
end

# %% Finding best predicting quantal level-k models.
function PQLK_pred(game::Game; λ=1.03, τ=0.5)
    level_0 = SimHeuristic([0.0,0.0,0.0])
    level_1 = SimHeuristic([0.0,0.0,λ])
    level_2 = SimHeuristic([0.0,0.0,λ, 0.0, 0.0, λ])
    π_dist = Distributions.Poisson(τ)
    p_vals = [Distributions.pdf(π_dist, i) for i in 0:2]
    p_vals ./ sum(p_vals)
    l0_pred = decide_probs(level_0, game)
    l1_pred = decide_probs(level_1, game)
    l2_pred = decide_probs(level_2, game)
    pred = l0_pred*p_vals[1] + l1_pred*p_vals[2] + l2_pred*p_vals[3]
end

function QLK_pred(game::Game; λ=2.3, α_0=0.07, α_1=0.64)
    level_0 = SimHeuristic([0.0,0.0,0.0])
    level_1 = SimHeuristic([0.0,0.0,λ])
    level_2 = SimHeuristic([0.0,0.0,λ, 0.0, 0.0, λ])
    α_2 = 1 - α_0 - α_1
    l0_pred = decide_probs(level_0, game)
    l1_pred = decide_probs(level_1, game)
    l2_pred = decide_probs(level_2, game)
    pred = l0_pred*α_0 + l1_pred*α_1 + l2_pred*α_2
end

function pred_loss(behavior_fun::Function, games, self_probs)
    pay = 0
    for i in eachindex(games)
        p = behavior_fun(games[i])
        pay +=  sum( (p - self_probs[i]).^2)
    end
    pay/length(games)
end

games = exp_games
row_plays = exp_row_plays
function wrap_pqlk(params)
    f = x -> PQLK_pred(x,;λ=params[1], τ=params[2])
    loss = pred_loss(f, games, row_plays)
end
wrap_pqlk([0.5, 1.03])

function wrap_qlk(params)
    f = x -> QLK_pred(x,;λ=params[1], α_0 =params[2], α_1=params[3])
    loss = pred_loss(f, games, row_plays)
end

res = Optim.minimizer(optimize(wrap_qlk, [0.3, 0.3, 0.3], BFGS(), Optim.Options(time_limit=60)))
wrap_qlk(res)

res
####################################################
# %% Finding optimal costs and share 1 vs 2 players
###################################################

function opt_cost_datasets(train_data_sets)
    train_idx = filter(i -> data_name_list[i] in train_data_sets, 1:length(data_name_list))
    train_games = exp_games[train_idx]
    train_row_plays = exp_row_plays[train_idx]
    train_col_plays = exp_col_plays[train_idx]

    @everywhere begin
        train_games = $train_games
        train_row_plays = $train_row_plays
        train_col_plays = $train_col_plays
    end

    costs = rand_costs(0.01, 0.2, 0.01, 0.4)
    res = costs_preds(costs, train_games, train_row_plays, train_col_plays)
    println(res)

    costs_perf = pmap(1:1000) do x
        try
            costs = rand_costs(0.01, 0.2, 0.01, 0.4)
            res = costs_preds(costs, train_games, train_row_plays, train_col_plays)
            println(res)
            if res != nothing
                return (res[1], costs, res[2:4]...)
            end
        catch err
            println(err)
        end
    end
    costs_perf = filter(x -> x !== nothing, costs_perf)
    println(costs_perf)
    sort!(costs_perf, by= x -> x[1], rev=false)
    costs_perf[1]
end

opt_combo = opt_cost_datasets(data_sets)


### Looking at optimal costs for different data_sets
opts = map(data_sets) do x
    opt_cost_datasets([x])
end



#%% Code for inspecting and comparing the different predictions
opt_s = deepcopy(opt_res1[1][1])
msd_s = deepcopy(msd_res[1][1])
ML_s = deepcopy(ML_res[1][1])
opt_s1 = opt_combo[4]
opt_s2 = opt_combo[5]
opt_α = opt_combo[3]
# s.h_list[1].γ = 1.
# # s.h_list[1].α = 2.
# s.h_list[1].λ = 5.
pred_loss(opt_s, exp_games, exp_row_plays)
pred_loss(opt_s1, opt_s2, opt_α, exp_games, exp_row_plays)

## See what indices pertain to a certain data set
idx_to_look_at = filter(i -> data_name_list[i] == data_sets[6], 1:length(data_name_list))

i = 80

println("Game: ", game_names[i])
println(exp_games[i])
println("MSD :", decide_probs(msd_s, exp_games[i]))
println("ML :", decide_probs(ML_s, exp_games[i]))
println("OPT :", decide_probs(opt_s, exp_games[i]))
println("PQLK: ", QLK_pred(exp_games[i]))
println("Opt 2 types: ", decide_probs(opt_s1, opt_s2, opt_α, exp_games[i]))
println("Actual: ", exp_row_plays[i])

### Compare performance of opt_comb over data_sets
println()
for data_set in data_sets
    idx = filter(i -> data_name_list[i] == data_set, 1:length(data_name_list))
    # println(idx)
    # println(pred_loss(opt_s1, opt_s2, opt_α, exp_games[idx], exp_row_plays[idx]))
    println("Opt_comb for ", data_set, " : ", pred_loss(opt_s1, opt_s2, opt_α, exp_games[idx], exp_row_plays[idx]))
    println("QLK for ", data_set, ": ", pred_loss(QLK_pred, exp_games[idx], exp_row_plays[idx]))
end


## Look at all predictions for a given data_set
idx = 7
idx_to_look_at = filter(i -> data_name_list[i] in data_sets[idx:idx], 1:length(data_name_list))
for i in idx_to_look_at
    println("Game: ", game_names[i])
    println(exp_games[i])
    println("MSD :", decide_probs(msd_s, exp_games[i]))
    println("ML :", decide_probs(ML_s, exp_games[i]))
    println("OPT :", decide_probs(opt_s, exp_games[i]))
    println("PQLK: ", QLK_pred(exp_games[i]))
    println("Opt 2 types: ", decide_probs(opt_s1, opt_s2, opt_α, exp_games[i]))
    println("Actual: ", exp_row_plays[i])
    println()
end


#%% Do locally optimized heuristics with global cognitive costs, performs bad


opt_costs = opts[2][2]
perf_per_dataset = map(data_sets) do x
    idx = filter(i -> data_name_list[i] == x, 1:length(data_name_list))
    res = costs_preds(opt_costs, exp_games[idx], exp_row_plays[idx], exp_col_plays[idx])
    res
end


### Find globally optimal costs for local optimization of heuristcs
α = 0.7
costs_perf = pmap(1:400) do x
    try
        costs = rand_costs(0.01, 0.2, 0.01, 0.4)
        perfs = map(data_sets) do x
            idx = filter(i -> data_name_list[i] == x, 1:length(data_name_list))
            res = costs_preds(costs, α, exp_games[idx], exp_row_plays[idx], exp_col_plays[idx])
            res
        end
        res = mean([x[1] for x in perfs])
        (res, costs)
    catch err
        pass
    end
end


sort!(costs_perf, by= x-> x[1])
