using Distributed
using Plots
import StatsBase: sample, std
using JSON
import Base: ==, hash


include("Heuristics.jl")

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

# Loading data

exp_games = normalize.(convert(Vector{Game}, games_from_json("data/games.json")))
exp_row_plays = plays_vec_from_json("data/rows_played.json")
exp_col_plays = plays_vec_from_json("data/cols_played.json")

opp_h = CacheHeuristic(transpose.(exp_games), exp_col_plays)
actual_h = CacheHeuristic(exp_games, exp_row_plays)
good_costs= Costs(0.47, 0.04, 0.42, 0.15, 2.4)

# %% Sharing data to all kernels
@everywhere begin
    include("Heuristics.jl")
end

@everywhere begin
    exp_games = $exp_games
    exp_col_plays = $exp_col_plays
    exp_row_plays = $exp_row_plays
    opp_h = CacheHeuristic(transpose.(exp_games), exp_col_plays)
    actual_h = CacheHeuristic(exp_games, exp_row_plays)
    good_costs= Costs(; α=0.47, λ=0.04, row=0.42, level=0.15, m_λ=2.4)
end


#############################################################
# %% Finding best predicting quantal level-k  and QCH models.
############################################################

qlk_h = QLK(0.07, 0.64, 2.3)
best_fit_qlk = fit_h!(qlk_h, exp_games, actual_h)
prediction_loss(qlk_h, exp_games, actual_h)

qch_h = QCH(0.07, 0.64, 2.3)
best_fit_qch = fit_h!(qch_h, exp_games, actual_h)
prediction_loss(qch_h, exp_games, actual_h)

l1_h = RowHeuristic(-0.3, 2.)
best_fit_l1 = fit_h!(l1_h, exp_games, actual_h)
prediction_loss(l1_h, exp_games, actual_h)
opt_l1 = optimize_h!(l1_h, exp_games, opp_h, good_costs)
prediction_loss(opt_l1, exp_games, actual_h)


####################################################
# %% Finding optimal costs and share 1 vs 2 players
###################################################
function fit_cost_opt_h(init_mh::MetaHeuristic, games, acutal_h, opp_h, n=64)
    @everywhere begin
        games = $games
        actual_h = $actual_h
        opp_h = $opp_h
        init_mh = deepcopy($init_mh)
    end

    costs_perf = pmap(1:n) do x
        try
            costs = rand_costs([0.4, 0.03, 0.5, 0.4, 1.], [0.401, 0.2, 0.501, 0.401, 5.])
            # costs = Costs(0.0383713, 0.408536, 0.0841975, 0.498675, 1.98782)
            # costs = Costs(0.02, 0.13, 0.001, 1.5, 10.)
            mh = deepcopy(init_mh)
            opt_mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(init_mh))
            # opt_mh = fit_h!(mh, games, actual_h, opp_h, costs; init_x = get_parameters(init_mh))
            # opt_mh = fit_prior!(opt_mh, games, actual_h, opp_h, costs)
            opt_mh = opt_prior!(opt_mh, games, opp_h, costs)
            res = prediction_loss(opt_mh, games, actual_h, opp_h, costs)
            return (res, costs, opt_mh)
        catch err
            println(err)
        end
    end
    costs_perf = filter(x -> x !== nothing, costs_perf)
    sort!(costs_perf, by= x -> x[1], rev=false)
    costs_perf
end


#%% Train and test sets
function std_hdist(h, games, opp_h, best_costs)
    stds =  map(1:length(h.prior)) do i
        std([h_distribution(h, game, opp_h, best_costs)[i] for game in exp_games])
    end
end

train_losses = Dict("level2_jm_cell" => [], "level2_jm" => [], "level2" => [], "qlk" => [])
test_losses = Dict("level2_jm_cell" => [], "level2_jm" => [], "level2" => [], "qlk" => [])
h_dists_dict = Dict("level2_jm_cell" => [], "level2_jm" => [], "level2" => [])
costs_dict = Dict("level2_jm_cell" => Costs[], "level2_jm" => Costs[], "level2" => Costs[])

for i in 1:5
    n_train = 100
    train_idxs = sample(1:length(exp_games), n_train; replace=false)
    test_idxs = setdiff(1:length(exp_games), train_idxs)
    sort!(train_idxs)
    sort!(test_idxs)
    train_games = exp_games[train_idxs]
    test_games = exp_games[test_idxs]


    # all_mh = MetaHeuristic([JointMax(2.), RowHeuristic(-.4, 2.), RandomHeuristic(), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)]), SimHeuristic([JointMax(2.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0.])
    level2_jm_mh = MetaHeuristic([JointMax(2.), RowHeuristic(-.4, 2.), RandomHeuristic(), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0.])
    level2_jm_cell_mh = MetaHeuristic([JointMax(2.), RowHeuristic(-.4, 2.), CellHeuristic(0.4, 2.), RandomHeuristic(), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0])
    level2_mh = MetaHeuristic([RowHeuristic(-0.4, 2.), RandomHeuristic(), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0.])

    costs_perfs_dict = Dict()

    # costs_perfs_dict["level2_jm_cell"] = fit_cost_opt_h(all_mh, train_games, actual_h, opp_h, 1*64)
    costs_perfs_dict["level2_jm"] = fit_cost_opt_h(level2_jm_mh, train_games, actual_h, opp_h, 4*64)
    costs_perfs_dict["level2_jm_cell"] = fit_cost_opt_h(level2_jm_cell_mh, train_games, actual_h, opp_h, 4*64)
    costs_perfs_dict["level2"] = fit_cost_opt_h(level2_mh, train_games, actual_h, opp_h, 4*64)
    qlk_h = QLK(0.07, 0.64, 2.3)
    best_fit_qlk = fit_h!(qlk_h, train_games, actual_h)
    for (name, results) in costs_perfs_dict
        best_perf = results[1][1]
        best_h = results[1][3]
        best_costs = results[1][2]
        println("---------------", name, "-----------")
        println("Prior: ", best_h.prior)
        println("Heuristic: ", best_h)
        println("Costs: ", best_costs)
        train_loss = prediction_loss(best_h, train_games, actual_h, opp_h, best_costs)
        test_loss = prediction_loss(best_h, test_games, actual_h, opp_h, best_costs)
        push!(train_losses[name], train_loss)
        push!(test_losses[name], test_loss)
        println(train_loss)
        println(test_loss)
        h_distss = [h_distribution(best_h, g, opp_h, best_costs) for g in exp_games]
        println(mean(h_distss))
        push!(h_dists_dict[name], mean(h_distss))
        push!(costs_dict[name], best_costs)
        println(std_hdist(best_h, exp_games, opp_h, best_costs))
    end
    qlk_train_loss =  prediction_loss(qlk_h, train_games, actual_h)
    qlk_test_loss =  prediction_loss(qlk_h, test_games, actual_h)
    push!(train_losses["qlk"], qlk_train_loss)
    push!(test_losses["qlk"], qlk_test_loss)
    println("QLK train: ",  qlk_train_loss, " test:", qlk_test_loss)
    println(qlk_h)
end




println()
for (key, val) in train_losses
    if length(val) > 0
        println(key,": ", mean(val))
    end
end

for (key, val) in test_losses
    if length(val) > 0
        println(key,": ", mean(val))
    end
end

for (key, val) in h_dists_dict
    if length(val) > 0
        println(key,": ", mean(val))
    end
end

for (key, val) in costs_dict
    if length(val) > 0
        println(key," c mean: ", mean(convert(Vector{Costs}, val)))
    end
end

for (key, val) in costs_dict
    if length(val) > 0
        println(key," c std: ", std(convert(Vector{Costs}, val)))
    end
end

costs_dict["level2"]

function mean(costs_vec::Vector{Costs})
    means = []
    for field in fieldnames(Costs)
        push!(means, mean([getfield(c, field) for c in costs_vec]))
    end
    means
end

function std(costs_vec::Vector{Costs})
    stds = []
    for field in fieldnames(Costs)
        push!(stds, std([getfield(c, field) for c in costs_vec]))
    end
    stds
end

vec_costs = convert(Vector{Costs}, costs_dict["level2"])
mean(vec_costs)
std(vec_costs)

# %% Comparing datasets
data_set_perfs = Dict()



for data_set in data_sets
    idxs = filter(x -> data_name_list[x] == data_set, 1:length(data_name_list))
    data_games = exp_games[idxs]
    all_mh = MetaHeuristic([NashHeuristic(2.), RowHeuristic(-.4, 2.), RandomHeuristic(), RowCellHeuristic(0.3, -0.2, 1.5), CellHeuristic(0.6, 1.), SimHeuristic([RowCellHeuristic(0.3, -0.2, 2.), RowCellHeuristic(0.3, -0.2, 2.)]), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
    no_α_mh = MetaHeuristic([NashHeuristic(2.), RowHeuristic(-.4, 2.), RandomHeuristic(), CellHeuristic(0.6, 1.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0.])
    # mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), CellHeuristic(0.6, 1.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0.])
    no_αγ_mh = MetaHeuristic([NashHeuristic(2.), RowMean(2.), RandomHeuristic(), SimHeuristic([RowMean(1.), RowMean(2.)])], [0., 0., 0., 0.])

    costs_perfs_dict = Dict()

    costs_perfs_dict["level2_jm_cell"] = fit_cost_opt_h(all_mh, data_games, actual_h, opp_h, 1*64)
    # costs_perfs_dict["level2_jm"] = fit_cost_opt_h(no_α_mh, data_games, actual_h, opp_h, 1*64)
    costs_perfs_dict["level2"] = fit_cost_opt_h(no_αγ_mh, data_games, actual_h, opp_h, 1*64)

    data_set_perfs[data_set] = []
    for (name, results) in costs_perfs_dict
        best_perf = results[1][1]
        best_h = results[1][3]
        best_costs = results[1][2]
        println("---------", data_set, "-------")
        println("Prior: ", best_h.prior)
        println("Heuristic: ", best_h)
        loss = prediction_loss(best_h, data_games, actual_h, opp_h, best_costs)
        println(loss)
        h_distss = [h_distribution(best_h, g, opp_h, best_costs) for g in exp_games]
        println(mean(h_distss))
        println(std_hdist(best_h, exp_games, opp_h, best_costs))
        push!(data_set_perfs[data_set], (name, loss))
    end
end

for (data_set, perfs) in data_set_perfs
    println(data_set, perfs)
end

[mean([perfs[i][2] for perfs in values(data_set_perfs)]) for i in 1:3]
