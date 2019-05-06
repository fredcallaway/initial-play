using Flux
using JSON
using CSV
using DataFrames
using SplitApplyCombine
using Random
using Glob
using Distributed
include("Heuristics.jl")


addprocs(Sys.CPU_THREADS)
@everywhere begin
    include("Heuristics.jl")
    include("rule_learning.jl")
    include("new_model.jl")
end


# %% ====================  ====================
Data = Array{Tuple{Game,Array{Float64,1}},1}
parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))

function load_data(file)::Data
    file
    df = CSV.read(file);
    row_plays = Data()
    col_plays = Data()
    for row in eachrow(df)
        row_game = json_to_game(row.row_game)
        row_game.row[1,2] += rand()*1e-7 # This can't be a constant number if we want to
        row_game.col[1,2] += rand()*1e-7 # separate behavior in comparison games in different treatments.
        row_play_dist = parse_play(row.row_play)
        col_game = transpose(row_game)
        col_play_dist = parse_play(row.col_play)
        push!(row_plays, (row_game, row_play_dist))
        push!(col_plays, (col_game, col_play_dist))
    end
    append!(row_plays, col_plays)
end

function load_treatment_data(treatment)
    files = glob("data/processed/$treatment/*_play_distributions.csv")
    data = vcat(map(load_data, files)...)
end

# %% ==================== Loss functions ====================
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)

loss_min(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)

function prediction_loss(model::MetaHeuristic, data::Data, costs)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play, empirical_play, costs)
end

function prediction_loss(model::Heuristic, data::Data, costs)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play)
end

function prediction_loss(model::Chain, data::Data, costs)
    loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
    loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
    loss_no_norm(data)
end

function prediction_loss(model::RuleLearning, data::Data, costs)
    rule_loss(model, data)
end



# %% ==================== Train test subsetting ====================
function comparison_indices(data)
    comparison_idx = [31, 34, 38, 41, 44, 50, 131, 137, 141, 144, 149]
    later_com = 50 .+ comparison_idx
    comparison_idx = [comparison_idx..., later_com...]
    comparison_idx = filter(x -> x <= length(data), comparison_idx)
    sort(comparison_idx)
end

function early_late_indices(data; n =30)
    train_idx = filter(x -> (x-1) % 50 < n, 1:length(data))
    test_idx = setdiff(1:length(data), train_idx)
    train_idx, test_idx
end

function leave_one_pop_out_indices(data, test_pop)
    test_idx = collect(1:100) .+ (test_pop-1)*100
    train_idx = setdiff(1:length(data), test_idx)
    train_idx, test_idx
end



function run_train_test(model, data::Data, train_idx::Vector{Int64}, test_idx::Vector{Int64}, mode::Symbol, cs::Union{Cost_Space, DeepCostSpace}, n::Int64; parallel=true)
    costs_vec  = rand(cs, n)
    map_over_vec = [(model, c, data[train_idx]) for c in costs_vec]
    mymap = parallel ? pmap : map
    make_f = mode == :fit ? make_fit : make_optimize
    res_model = mymap(x -> make_f(x[1], x[2])(x[3]), map_over_vec)
    perfs = map(i -> prediction_loss(res_model[i], data[train_idx], costs_vec[i]), 1:n)
    perfs = map(x -> isnan(x) ? Inf : x, perfs)
    best_idx = argmin(perfs)
    model = res_model[best_idx]
    costs = costs_vec[best_idx]
    comp_idx = comparison_indices(data)
    res_dict = Dict()
    if typeof(model) != RuleLearning
        res_dict = Dict(
            :train_loss => prediction_loss(model, data[train_idx], costs),
            :test_loss => prediction_loss(model, data[test_idx], costs),
            :test_loss_nocomp => prediction_loss(model, data[setdiff(test_idx, comp_idx)], costs),
            :comparison_loss => prediction_loss(model, data[comp_idx], costs),
            :model => model,
            :costs => costs,
        )
    else
        res_dict = Dict(
            :train_loss => prediction_loss(model, data[train_idx], costs),
            :test_loss => prediction_loss(model, data[test_idx], costs),
            :model => model,
            :costs => costs,
        )
    end
    return res_dict
end

function pop_cross_validation(model, data::Data, mode::Symbol, cs::Union{Cost_Space, DeepCostSpace}, n::Int64; parallel=true)
    test_losses = []
    for i in 1:Int(length(data)/100)
        train_idx, test_idx = leave_one_pop_out_indices(data, i)
        res = run_train_test(model, data, train_idx, test_idx, mode, cs, n; parallel=parallel)
        push!(test_losses, res[:test_loss])
        println(res[:test_loss])
    end
    return test_losses
end




# %% =============== Load data  =======================
pos_data = load_treatment_data("positive")
neg_data = load_treatment_data("negative")
all_data = [pos_data; neg_data]

comp_idx = comparison_indices(pos_data)
train_idx, test_idx = early_late_indices(pos_data)
train_idx, test_idx = leave_one_pop_out_indices(pos_data, 1)


# %% Setup and run
mh_base = MetaHeuristic([RandomHeuristic(), JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0.]);
qch_base = QCH(0.3, 0.3, 1.)
cs = Cost_Space((0.1, 0.2), (0.1, 0.2), (0., 0.2), (0.7,1.5))

deep_base = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))
deep_cs = DeepCostSpace((0.001,0.01), (0.2, 0.5), (0.0, 0.3))

rl_base = RuleLearning(deepcopy(mh_base), 1., 1., rand(cs))

fit_mh_neg_res = run_train_test(mh_base, neg_data, train_idx, test_idx, :fit, cs, 8*5)
fit_qch_neg_res = run_train_test(qch, neg_data, train_idx, test_idx, :fit, cs, 8*5)
fit_deep_neg_res = run_train_test(deep_base, neg_data, train_idx, test_idx, :fit, deep_cs, 5)

rl_train_idx, rl_test_idx = leave_one_pop_out_indices(all_data, 1)
fit_rl_all_res = run_train_test(rl_base, all_data, train_idx, test_idx, :fit, cs, 5)

rl_losses = pop_cross_validation(rl_base, all_data, :fit, cs, 5)
rl_losses_pos = pop_cross_validation(rl_base, pos_data, :fit, cs, 5)
rl_losses_neg = pop_cross_validation(rl_base, neg_data, :fit, cs, 5)
mh_losses_pos = pop_cross_validation(mh_base, pos_data, :fit, cs, 5)
mh_losses_neg = pop_cross_validation(mh_base, neg_data, :fit, cs, 5)
qch_losses_pos = pop_cross_validation(qch_base, pos_data, :fit, cs, 5)
qch_losses_neg = pop_cross_validation(qch_base, neg_data, :fit, cs, 5)
deep_losses_pos = pop_cross_validation(deep_base, pos_data, :fit, deep_cs, 5)
deep_losses_neg = pop_cross_validation(deep_base, neg_data, :fit, deep_cs, 5)


#= Analysis Plan

1. Implement Train 30 Test 20
2. Implement LOOCV at the population level
3. Fit and optimize MetaHeuritic
    - Fit costs when fitting parameters
4. Fit and optimize Deep Heuristic
5. Fit RuleLearning

=#
