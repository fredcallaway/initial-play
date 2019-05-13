using Flux
using JSON
using CSV
using DataFrames
using SplitApplyCombine
using Random
using Glob
using Distributed
using BSON
using Serialization
include("Heuristics.jl")


addprocs(Sys.CPU_THREADS)
@everywhere begin
    include("Heuristics.jl")
    include("rule_learning.jl")
    include("new_model.jl")
end


# %% ====================  ====================
@everywhere Data = Array{Tuple{Game,Array{Float64,1}},1}
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

function prediction_loss(model::MetaHeuristic, in_data::Data, idx, costs)
    data = in_data[idx]
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play, empirical_play, costs)
end

function prediction_loss(model::Heuristic, in_data::Data, idx, costs)
    data = in_data[idx]
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play)
end

function prediction_loss(model::Chain, in_data::Data, idx, costs)
    data = in_data[idx]
    loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
    loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
    loss_no_norm(data).data
end

function prediction_loss(model::RuleLearning, in_data::Data, idx, costs)
    rule_loss_idx(model, in_data, idx)
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


function run_train_test(model, neg_data::Data, pos_data::Data, train_idx::Vector{Int64}, test_idx::Vector{Int64}, mode::Symbol, costs_vec::Union{Vector{Costs}, Vector{DeepCosts}}; parallel=true)
    mymap = parallel ? pmap : map
    n = length(costs_vec)
    train = Dict(
        :fit => fit_model,
        :opt => optimize_model
    )[mode]
    res_model_dict = Dict()
    for (treat, data) in zip([:negative, :positive], [neg_data, pos_data])
        res_model = mymap(costs_vec) do costs
            train(model, data, train_idx, costs)
        end
        res_model_dict[treat] = res_model
    end

    perfs = map(1:n) do i
        neg_train_loss = prediction_loss(res_model_dict[:negative][i], neg_data, train_idx, costs_vec[i])
        pos_train_loss = prediction_loss(res_model_dict[:positive][i], pos_data, train_idx, costs_vec[i])
        neg_train_loss + pos_train_loss
    end
    perfs = map(x -> isnan(x) ? Inf : x, perfs)

    best_idx = argmin(perfs)
    pos_model = res_model_dict[:positive][best_idx]
    neg_model = res_model_dict[:negative][best_idx]
    costs = costs_vec[best_idx]
    comp_idx = comparison_indices(pos_data)
    res_dict = Dict(
        :pos_train_loss => prediction_loss(pos_model, pos_data, train_idx, costs),
        :pos_test_loss => prediction_loss(pos_model, pos_data, test_idx, costs),
        :pos_test_loss_nocomp => prediction_loss(pos_model, pos_data, setdiff(test_idx, comp_idx), costs),
        :pos_comparison_loss => prediction_loss(pos_model, pos_data, comp_idx, costs),
        :pos_model => pos_model,
        :neg_train_loss => prediction_loss(neg_model, neg_data, train_idx, costs),
        :neg_test_loss => prediction_loss(neg_model, neg_data, test_idx, costs),
        :neg_test_loss_nocomp => prediction_loss(neg_model, neg_data, setdiff(test_idx, comp_idx), costs),
        :neg_comparison_loss => prediction_loss(neg_model, neg_data, comp_idx, costs),
        :neg_model => neg_model,
        :costs => costs)
    println(res_dict[:pos_test_loss], " - ", res_dict[:neg_test_loss])
    return res_dict
end

function pop_cross_validation(model, neg_data::Data, pos_data::Data, mode::Symbol, costs_vec::Union{Vector{Costs}, Vector{DeepCosts}}; parallel=true)
    test_losses = []
    for i in 1:Int(length(pos_data)/100)
        train_idx, test_idx = leave_one_pop_out_indices(pos_data, i)
        res = run_train_test(model, neg_data, pos_data, train_idx, test_idx, mode, costs_vec; parallel=parallel)
        push!(test_losses, (res[:pos_test_loss], res[:neg_train_loss]))
    end
    return test_losses
end




# %% =============== Load data  =======================
pos_data = load_treatment_data("positive")
neg_data = load_treatment_data("negative")
all_data = [pos_data; neg_data]



comp_idx = comparison_indices(pos_data)
train_idx, test_idx = early_late_indices(pos_data)


# %% Setup and run
mh_base = MetaHeuristic([RandomHeuristic(), JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0.]);
qch_base = QCH(0.3, 0.3, 1.)
cs = Cost_Space((0.5, 0.3), (0.5, 0.3), (0., 0.2), (0.5,2.5))

deep_base = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))
deep_cs = DeepCostSpace((0.001,0.01), (0.2, 0.5), (0.0, 0.3))

rl_base = RuleLearning(deepcopy(mh_base), 1., 1., rand(cs))


#%% Run
n_runs = 64*4
mh_costs_vec  = rand(cs, n_runs)
deep_costs_vec  = rand(deep_cs, n_runs)


res_dict = Dict()
save_file = "saved_objects/res_dict2"
res_dict[:fit_rl] = run_train_test(rl_base, neg_data, pos_data, train_idx, test_idx, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_mh] = run_train_test(mh_base, neg_data, pos_data, train_idx, test_idx, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_qch] = run_train_test(qch_base, neg_data, pos_data, train_idx, test_idx, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_deep] = run_train_test(deep_base, neg_data, pos_data, train_idx, test_idx, :fit, deep_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_mh] = run_train_test(mh_base, neg_data, pos_data, train_idx, test_idx, :opt, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_qch] = run_train_test(qch_base, neg_data, pos_data, train_idx, test_idx, :opt, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_deep] = run_train_test(deep_base, neg_data, pos_data, train_idx, test_idx, :opt, deep_costs_vec)
serialize(save_file, res_dict)

res_dict[:fit_rl_cv] = pop_cross_validation(rl_base, neg_data, pos_data, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_mh_cv] = pop_cross_validation(mh_base, neg_data, pos_data, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_qch_cv] = pop_cross_validation(qch_base, neg_data, pos_data, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_deep_cv] = pop_cross_validation(deep_base, neg_data, pos_data, :fit, deep_costs_vec)
serialize(save_file, res_dict)

res_dict[:opt_mh_cv] = pop_cross_validation(mh_base, neg_data, pos_data, :opt, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_qch_cv] = pop_cross_validation(qch_base, neg_data, pos_data, :opt, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_deep_cv] = pop_cross_validation(deep_base, neg_data, pos_data, :opt, deep_costs_vec)
serialize(save_file, res_dict)

######################################################
#%% Generate Data Frame
######################################################
res_dict = deserialize("saved_objects/res_dict")

res_df = DataFrame()
first_last_symbols = [:fit_qch, :opt_qch, :fit_mh, :opt_mh, :fit_deep, :opt_deep, :fit_rl]

function cat_syms(a, b; sep="_")
    Symbol(String(a)*sep*String(b))
end

for (treatment, data) in zip(["Competing", "Common"], [neg_data, pos_data])
    for (data_type, idxs) in zip(["all", "train", "test", "comparison"], [collect(1:length(pos_data)), train_idx, test_idx, comp_idx])
        row_dict = Dict{Any, Any}(:data_type => data_type, :treatment => treatment)
        row_dict[:random] = rand_loss(data[idxs])
        row_dict[:minimum] = min_loss(data[idxs])
        for sym in first_last_symbols
            for (model, treat_data) in zip([res_dict[sym][:neg_model], res_dict[sym][:pos_model]], ["neg", "pos"])
                row_dict[cat_syms(sym,treat_data)] = prediction_loss(model, data, idxs, res_dict[sym][:costs])
            end
        end
        if length(names(res_df)) == 0
            res_df = DataFrame(row_dict)
        else
            push!(res_df, row_dict)
        end
    end
end

serialize("saved_objects/res_df",res_df)


######################################################
#%%
######################################################



using Plots
using StatsPlots

pyplot()

keys_to_plot = [:random, :fit_qch_neg, :fit_qch_pos, :opt_qch_neg, :opt_qch_pos, :fit_mh_neg, :fit_mh_pos, :opt_mh_neg, :opt_mh_pos, :fit_deep_neg,  :fit_deep_pos, :opt_deep_neg, :opt_deep_pos, :minimum]

plots_vec = []
for data_type in ["train", "test", "comparison"], treat in ["Competing", "Common"]
    vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), keys_to_plot]))
    ctg = [repeat(["competing"], 6)..., repeat(["commmon"], 6)...]
    nam = [repeat(["fit qch", "opt qch", "fit mh", "opt mh", "fit deep", "opt deep"],2)...]
    bar_vals = hcat(transpose(reshape(vals[2:(end-1)], (2,6))))
    plt = groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*" interest "*data_type*" games", ylims=(0,1.3))
    plot!([vals[1]], linetype=:hline, width=2, label="random loss", color=:grey)
    plot!([vals[14]], linetype=:hline, width=2, label="min loss", color=:black)
    push!(plots_vec, plt)
end

length(plots_vec)
plot(plots_vec..., layout=(3,2), size=(794,1123))

savefig("../overleaf/figs/loglikelihoods.png")

res_df

#= Analysis Plan

1. Implement Train 30 Test 20
2. Implement LOOCV at the population level
3. Fit and optimize MetaHeuritic
    - Fit costs when fitting parameters
4. Fit and optimize Deep Heuristic
5. Fit RuleLearning

=#
