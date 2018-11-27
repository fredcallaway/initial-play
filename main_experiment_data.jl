using Distributed
using Plots
import StatsBase: sample
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
    good_costs= Costs(0.47, 0.04, 0.42, 0.15, 2.4)
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
            costs = rand_costs([0.05, 0.04, 0.05, 0.05, 1.], [1., 0.5, 0.7, 0.7, 10.])
            # costs = Costs(0.0383713, 0.408536, 0.0841975, 0.498675, 1.98782)
            # costs = Costs(0.02, 0.13, 0.001, 1.5, 10.)
            mh = deepcopy(init_mh)
            opt_mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(init_mh))
            # opt_mh = fit_h!(mh, games, actual_h, opp_h, costs; init_x = get_parameters(init_mh))
            opt_mh = fit_prior!(opt_mh, games, actual_h, opp_h, costs)
            # opt_mh = opt_prior!(opt_mh, games, opp_h, costs)
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


mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), RowCellHeuristic(0.3, -0.2, 1.5), CellHeuristic(0.6, 1.), SimHeuristic([RowCellHeuristic(0.3, -0.2, 2.), RowCellHeuristic(0.3, -0.2, 2.)]), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0.])
mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0.])
# mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), CellHeuristic(0.6, 1.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0.])
mh = MetaHeuristic([RowMean(2.), RandomHeuristic(), SimHeuristic([RowMean(1.), RowMean(2.)])], [0., 0., 0.])
costs_perf = fit_cost_opt_h(mh, exp_games, actual_h, opp_h, 5*64)

costs_perf
[h_distribution(costs_perf[1][3], game, opp_h, costs_perf[1][2]) for game in exp_games]


for x in costs_perf[1:30]
    h_dists= [h_distribution(x[3], game, opp_h, x[2]) for game in exp_games]
    println(mean(h_dists))
end

Sys.CPU_THREADS

#%% Train and test sets
n_train = 100
train_idxs = sample(1:length(exp_games), n_train; replace=false)
test_idxs = setdiff(1:length(exp_games), train_idxs)
sort!(train_idxs)
sort!(test_idxs)
train_games = exp_games[train_idxs]
test_games = exp_games[test_idxs]

all_mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), RowCellHeuristic(0.3, -0.2, 1.5), CellHeuristic(0.6, 1.), SimHeuristic([RowCellHeuristic(0.3, -0.2, 2.), RowCellHeuristic(0.3, -0.2, 2.)]), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0.])
no_α_mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0.])
# mh = MetaHeuristic([RowHeuristic(-.4, 2.), RandomHeuristic(), CellHeuristic(0.6, 1.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0.])
no_αγ_mh = MetaHeuristic([RowMean(2.), RandomHeuristic(), SimHeuristic([RowMean(1.), RowMean(2.)])], [0., 0., 0.])
costs_perf_all = fit_cost_opt_h(all_mh, train_games, actual_h, opp_h, 2*64)
costs_perf_nα = fit_cost_opt_h(no_α_mh, train_games, actual_h, opp_h, 2*64)
costs_perf_nαγ = fit_cost_opt_h(no_αγ_mh, train_games, actual_h, opp_h, 2*64)


for perfs in [costs_perf_all, costs_perf_nα, costs_perf_nαγ]
    best_perf = perfs[1][1]
    best_h = perfs[1][3]
    best_costs = perfs[1][2]
    println("Prior: ", best_h.prior)
    println("Heuristic: ", best_h)
    println(prediction_loss(best_h, train_games, actual_h, opp_h, best_costs))
    println(prediction_loss(best_h, test_games, actual_h, opp_h, best_costs))
    h_dists= [h_distribution(best_h, game, opp_h, best_costs) for game in exp_games]
    println(mean(h_dists))
end
