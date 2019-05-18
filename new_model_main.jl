using Flux
using JSON
using CSV
using DataFrames
using DataFramesMeta
using SplitApplyCombine
using Random
using Glob
using Distributed
using BSON
using Serialization
using StatsBase
using Statistics
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
    test_idx = setdiff(test_idx, comparison_indices(data))
    train_idx = setdiff(train_idx, comparison_indices(data))
    train_idx, test_idx
end

function leave_one_pop_out_indices(data, test_pop)
    test_idx = collect(1:100) .+ (test_pop-1)*100
    train_idx = setdiff(1:length(data), test_idx)
    test_idx = setdiff(test_idx, comparison_indices(data))
    train_idx = setdiff(train_idx, comparison_indices(data))
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
        res[:train_idx] = train_idx
        res[:test_idx] = test_idx
        res[:pop_out] = i
        push!(test_losses, res)
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
mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
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
# res_dict[:fit_rl] = run_train_test(rl_base, neg_data, pos_data, train_idx, test_idx, :fit, mh_costs_vec)
# serialize(save_file, res_dict)
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

# res_dict[:fit_rl_cv] = pop_cross_validation(rl_base, neg_data, pos_data, :fit, mh_costs_vec)
# serialize(save_file, res_dict)
res_dict[:fit_qch_cv] = pop_cross_validation(qch_base, neg_data, pos_data, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_qch_cv] = pop_cross_validation(qch_base, neg_data, pos_data, :opt, mh_costs_vec)
serialize(save_file, res_dict)

res_dict[:fit_mh_cv] = pop_cross_validation(mh_base, neg_data, pos_data, :fit, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:fit_deep_cv] = pop_cross_validation(deep_base, neg_data, pos_data, :fit, deep_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_mh_cv] = pop_cross_validation(mh_base, neg_data, pos_data, :opt, mh_costs_vec)
serialize(save_file, res_dict)
res_dict[:opt_deep_cv] = pop_cross_validation(deep_base, neg_data, pos_data, :opt, deep_costs_vec)
serialize(save_file, res_dict)

######################################################
#%% Generate Data Frame For Leave One Population Out
######################################################
res_dict = deserialize("saved_objects/res_dict2")

res_dict[:fit_mh_cv][1]

cv_symbols = [:fit_mh_cv, :opt_mh_cv, :fit_deep_cv, :opt_deep_cv]


function cat_syms(a, b; sep="_")
    Symbol(String(a)*sep*String(b))
end
function cat_syms(a, b, c; sep="_")
    Symbol(String(a)*sep*String(b)*sep*String(c))
end


res_df = DataFrame()
for (treatment, data) in zip(["Competing", "Common"], [neg_data, pos_data])
    train_dict = Dict{Symbol, Any}(:treatment => treatment, :data_type => "train")
    test_dict = Dict{Symbol, Any}(:treatment => treatment, :data_type => "test")
    train_dict[:random] = rand_loss(data)
    train_dict[:minimum] = min_loss(data)
    test_dict[:random] = rand_loss(data)
    test_dict[:minimum] = min_loss(data)
    for sym in cv_symbols
        for treat_data in ["neg", "pos"]
            train_dict[cat_syms(sym,treat_data)] = []
            test_dict[cat_syms(sym,treat_data)] = []
        end
    end
    for pop_i in 1:2
        train_idx, test_idx = leave_one_pop_out_indices(data, pop_i)
        for sym in cv_symbols
            for (model, treat_data) in zip([res_dict[sym][pop_i][:neg_model], res_dict[sym][pop_i][:pos_model]], ["neg", "pos"])
                push!(train_dict[cat_syms(sym,treat_data)], prediction_loss(model, data, train_idx, res_dict[sym][pop_i][:costs]))
                push!(test_dict[cat_syms(sym,treat_data)], prediction_loss(model, data, test_idx, res_dict[sym][pop_i][:costs]))
            end
        end
    end

    for sym in cv_symbols
        for treat_data in ["neg", "pos"]
            train_dict[cat_syms(sym,treat_data,"std")] = Statistics.std(train_dict[cat_syms(sym,treat_data)])
            train_dict[cat_syms(sym,treat_data,"mean")] = mean(train_dict[cat_syms(sym,treat_data)])
            test_dict[cat_syms(sym,treat_data,"std")] = Statistics.std(test_dict[cat_syms(sym,treat_data)])
            test_dict[cat_syms(sym,treat_data,"mean")] = mean(test_dict[cat_syms(sym,treat_data)])
        end
    end
    if length(names(res_df)) == 0
        res_df = DataFrame(Dict(k=>typeof(v)[] for (k,v) in train_dict))
    end
    push!(res_df, train_dict)
    push!(res_df, test_dict)
end


res_df

CSV.write("results/LOO_results_to_plot.csv", res_df)
serialize("saved_objects/res_df_cv",res_df)

#%% Plot
using Plots
using StatsPlots

pyplot()

keys_to_plot = [:random, :fit_mh_cv_neg_mean, :fit_mh_cv_pos_mean, :opt_mh_cv_neg_mean, :opt_mh_cv_pos_mean, :fit_deep_cv_neg_mean,  :fit_deep_cv_pos_mean, :opt_deep_cv_neg_mean, :opt_deep_cv_pos_mean, :minimum]

plots_vec = []
for data_type in ["train", "test"], treat in ["Competing", "Common"]
    data_type = "train"
    tret = "Competing"
    vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), keys_to_plot]))
    ctg = [repeat(["competing"], 4)..., repeat(["commmon"], 4)...]
    nam = [repeat(["fit mh", "opt mh", "fit deep", "opt deep"],2)...]
    bar_vals = hcat(transpose(reshape(vals[2:(end-1)], (2,4))))
    plt = groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*" interest "*data_type*" games", ylims=(0,1.3))
    plot!([vals[1]], linetype=:hline, width=2, label="random loss", color=:grey)
    plot!([vals[10]], linetype=:hline, width=2, label="min loss", color=:black)
    push!(plots_vec, plt)
end

length(plots_vec)
plot(plots_vec..., layout=(2,2), size=(794,1000))

savefig("../overleaf/figs/loglikelihoods.png")


######################################################
#%% Generate payoff for different heuristics DF
######################################################
# Generate res for full data set
n_runs = 64
mh_costs_vec  = rand(cs, n_runs)
deep_costs_vec  = rand(deep_cs, n_runs)

full_data_res_dict = Dict()
save_file_full = "saved_objects/full_data_res_dict"
all_idx = collect(1:length(pos_data))
full_data_res_dict[:fit_mh] = run_train_test(mh_base, neg_data, pos_data, all_idx, all_idx, :fit, mh_costs_vec)
serialize(save_file_full, full_data_res_dict)
full_data_res_dict[:fit_deep] = run_train_test(deep_base, neg_data, pos_data, all_idx, all_idx, :fit, deep_costs_vec)
serialize(save_file_full, full_data_res_dict)
full_data_res_dict[:opt_mh] = run_train_test(mh_base, neg_data, pos_data, all_idx, all_idx, :opt, mh_costs_vec)
serialize(save_file_full, full_data_res_dict)
full_data_res_dict[:opt_deep] = run_train_test(deep_base, neg_data, pos_data, all_idx, all_idx, :opt, deep_costs_vec)
serialize(save_file_full, full_data_res_dict)



# Generate DF
full_data_res_dict_load = deserialize(save_file_full)


function expected_payoff(h::Chain, opponent::Heuristic, g::Game)
    p = h(g)
    p_opp = play_distribution(opponent, transpose(g))
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row).data
end

function expected_payoff(h::Chain, opponent::Heuristic, g::Game, costs)
    expected_payoff(h, opponent, g)
end


function max_payoff(opponent::Heuristic, g::Game)
    p_opp = play_distribution(opponent, transpose(g))
    maximum(sum(g.row .* p_opp', dims=2))
end

function rand_payoff(opponent::Heuristic, g::Game)
    p_opp = play_distribution(opponent, transpose(g))
    mean(sum(g.row .* p_opp', dims=2))
end

pos_games, pos_plays = invert(pos_data)
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);
neg_games, neg_plays = invert(neg_data)
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);

m_opt_pos = full_data_res_dict[:opt_deep][:pos_model]
m_opt_neg = full_data_res_dict[:opt_deep][:neg_model]
m_fit_pos = full_data_res_dict[:fit_deep][:pos_model]
m_fit_neg = full_data_res_dict[:fit_deep][:neg_model]
h_opt_pos = full_data_res_dict[:opt_mh][:pos_model]
h_opt_neg = full_data_res_dict[:opt_mh][:neg_model]
h_fit_pos = full_data_res_dict[:fit_mh][:pos_model]
h_fit_neg = full_data_res_dict[:fit_mh][:neg_model]
costs_opt = full_data_res_dict[:opt_mh][:costs]
costs_fit = full_data_res_dict[:fit_mh][:costs]




payoff_df = DataFrame()

for fit_type in [:fit_deep, :fit_mh]
    for treat in [:neg, :pos]
        row_dict = Dict()
        for (data_name, data) in zip([:Common, :Competing], [pos_data, neg_data])
            m_sym = cat_syms(treat, :model)
            m = full_data_res_dict[fit_type][m_sym]
            costs = full_data_res_dict[fit_type][:costs]
            games, plays = invert(data);
            empirical_play = CacheHeuristic(games, plays);
            row_dict[:model] = cat_syms(fit_type, treat)
            row_dict[data_name] = mean([expected_payoff(m, empirical_play, g, costs) for g in games])
        end
        if length(names(payoff_df)) == 0
            payoff_df = DataFrame(row_dict)
        else
            push!(payoff_df, row_dict)
        end
    end
end

rand_dict = Dict{Symbol, Any}(:model => :Random)
max_dict = Dict{Symbol, Any}(:model => :Maximum)
actual_dict = Dict{Symbol, Any}(:model => :Actual)
for (data_name, data) in zip([:Common, :Competing], [pos_data, neg_data])
    m_sym = cat_syms(treat, :model)
    m = full_data_res_dict[fit_type][m_sym]
    costs = full_data_res_dict[fit_type][:costs]
    games, plays = invert(data);
    empirical_play = CacheHeuristic(games, plays);
    max_dict[data_name] = mean([max_payoff(empirical_play, g) for g in games])
    rand_dict[data_name] = mean([rand_payoff(empirical_play, g) for g in games])
    actual_dict[data_name] = mean([expected_payoff(empirical_play, empirical_play, g) for g in games])
end
push!(payoff_df, rand_dict)
push!(payoff_df, actual_dict)
push!(payoff_df, max_dict)

payoff_df = payoff_df[:, [:model, :Common, :Competing]]

# Print table
using LaTeXTabulars
using LaTeXStrings

function gen_row(namn)
    comm = round(@where(payoff_df, :model .== namn).Common[1], digits=2)
    comp = round(@where(payoff_df, :model .== namn).Competing[1], digits=2)
    [replace(String(namn), "_" => " ") comm comp]
end
name_list = filter(x -> !(x in [:Maximum, :Actual, :Random]), payoff_df.model)
res_mat = reshape(gen_row(name_list[1]), (1,3))
for namn in name_list[2:end]
    res_mat = vcat(res_mat, gen_row(namn))
end
println(res_mat)

latex_tabular("./../overleaf/tex_snippets/payoff_new.tex",
              Tabular("lcc"),
              [Rule(:top),
               ["", "Common interest games", "Competing interest games"],
               Rule(:mid),
               Rule(),           # a nice \hline to make it ugly
               res_mat,
               # vec(gen_row(:Random)), # ragged!
               vec(gen_row(:Actual)), # ragged!
               Rule(),
               vec(gen_row(:Maximum)),
               Rule(:bottom)])

######################################################
# %% Tables for estimated heuristics
######################################################
fit_mh_pos = deepcopy(full_data_res_dict[:fit_mh][:pos_model])
fit_mh_neg = deepcopy(full_data_res_dict[:fit_mh][:neg_model])
opt_mh_pos = deepcopy(full_data_res_dict[:opt_mh][:pos_model])
opt_mh_neg = deepcopy(full_data_res_dict[:opt_mh][:neg_model])

opt_costs = full_data_res_dict[:opt_mh][:costs]
fit_costs = full_data_res_dict[:fit_mh][:costs]

pos_games, pos_plays = invert(pos_data);
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);
neg_games, neg_plays = invert(neg_data);
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);




opt_h_dists_pos = [h_distribution(opt_mh_pos, g, pos_empirical_play, opt_costs) for g in pos_games];
avg_opt_h_dist_pos = mean(opt_h_dists_pos)
opt_h_dists_neg = [h_distribution(opt_mh_neg, g, neg_empirical_play, opt_costs) for g in neg_games];
avg_opt_h_dist_neg = mean(opt_h_dists_neg)
fit_h_dists_pos = [h_distribution(fit_mh_pos, g, pos_empirical_play, fit_costs) for g in pos_games];
avg_fit_h_dist_pos = mean(fit_h_dists_pos)
fit_h_dists_neg = [h_distribution(fit_mh_neg, g, neg_empirical_play, fit_costs) for g in neg_games];
avg_fit_h_dist_neg = mean(fit_h_dists_neg)

estimated_mh_table = """
\\begin{tabular}{@{}l l c  c  c }
% & & \\multicolumn{3}{c}{Estimated} \\\\
  \\multicolumn{5}{c}{\\textbf{Estimated Meta Heuristic}}   \\\\ \\cline{1-5}
& & \\multicolumn{1}{ c}{Jointmax}  & Row Heuristic & \\multicolumn{1}{c}{Sim}  \\\\
& & \\multicolumn{1}{ c}{\$\\varphi\$}  & \$\\gamma, \\varphi\$ & \\multicolumn{1}{c}{\$(\\gamma, \\varphi), \\varphi\$}  \\\\ \\cline{1-5}
\\multicolumn{1}{l|}{\\multirow{2}{*}{Common interests}} & \\multicolumn{1}{l|}{Params} & $(round(fit_mh_pos.h_list[1].λ, digits=2)) & $(round(fit_mh_pos.h_list[2].γ, digits=2)), $(round(fit_mh_pos.h_list[2].λ, digits=2)) & ($(round(fit_mh_pos.h_list[3].h_list[1].γ, digits=2)), $(round(fit_mh_pos.h_list[3].h_list[1].λ, digits=2))), $(round(fit_mh_pos.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_fit_h_dist_pos[1]*100)) \\% & $(round(Int, avg_fit_h_dist_pos[2]*100)) \\% & $(round(Int, avg_fit_h_dist_pos[3]*100)) \\% \\\\
\\cline{1-5}\\multicolumn{1}{l|}{\\multirow{2}{*}{Competing interests}} &\\multicolumn{1}{l|}{Params} & $(round(fit_mh_neg.h_list[1].λ, digits=2)) & $(round(fit_mh_neg.h_list[2].γ, digits=2)), $(round(fit_mh_neg.h_list[2].λ, digits=2)) & ($(round(fit_mh_neg.h_list[3].h_list[1].γ, digits=2)), $(round(fit_mh_neg.h_list[3].h_list[1].λ, digits=2))), $(round(fit_mh_neg.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_fit_h_dist_neg[1]*100)) \\% & $(round(Int, avg_fit_h_dist_neg[2]*100)) \\% & $(round(Int, avg_fit_h_dist_neg[3]*100)) \\% \\\\ \\cline{1-5}
\\end{tabular}
"""
println(estimated_mh_table)

open("../overleaf/tex_snippets/estim_meta_heuristics.tex","w") do f
    write(f, estimated_mh_table)
end

optimal_mh_table = """
\\begin{tabular}{@{}l l c  c  c }
% & & \\multicolumn{3}{c}{Estimated} \\\\
  \\multicolumn{5}{c}{\\textbf{Optimal Meta Heuristic}}   \\\\ \\cline{1-5}
& & \\multicolumn{1}{ c}{Jointmax}  & Row Heuristic & \\multicolumn{1}{c}{Sim}  \\\\
& & \\multicolumn{1}{ c}{\$\\varphi\$}  & \$\\gamma, \\varphi\$ & \\multicolumn{1}{c}{\$(\\gamma, \\varphi), \\varphi\$}  \\\\ \\cline{1-5}
\\multicolumn{1}{l|}{\\multirow{2}{*}{Common interests}} & \\multicolumn{1}{l|}{Params} & $(round(opt_mh_pos.h_list[1].λ, digits=2)) & $(round(opt_mh_pos.h_list[2].γ, digits=2)), $(round(opt_mh_pos.h_list[2].λ, digits=2)) & ($(round(opt_mh_pos.h_list[3].h_list[1].γ, digits=2)), $(round(opt_mh_pos.h_list[3].h_list[1].λ, digits=2))), $(round(opt_mh_pos.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_opt_h_dist_pos[1]*100)) \\% & $(round(Int, avg_opt_h_dist_pos[2]*100)) \\% & $(round(Int, avg_opt_h_dist_pos[3]*100)) \\% \\\\
\\cline{1-5}\\multicolumn{1}{l|}{\\multirow{2}{*}{Competing interests}} &\\multicolumn{1}{l|}{Params} & $(round(opt_mh_neg.h_list[1].λ, digits=2)) & $(round(opt_mh_neg.h_list[2].γ, digits=2)), $(round(opt_mh_neg.h_list[2].λ, digits=2)) & ($(round(opt_mh_neg.h_list[3].h_list[1].γ, digits=2)), $(round(opt_mh_neg.h_list[3].h_list[1].λ, digits=2))), $(round(opt_mh_neg.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_opt_h_dist_neg[1]*100)) \\% & $(round(Int, avg_opt_h_dist_neg[2]*100)) \\% & $(round(Int, avg_opt_h_dist_neg[3]*100)) \\% \\\\ \\cline{1-5}
\\end{tabular}
"""
println(optimal_mh_table)


open("../overleaf/tex_snippets/opt_meta_heuristics.tex","w") do f
    write(f, optimal_mh_table)
end



###########################################################################
#%% Chi-square test
###########################################################################
using HypothesisTests

pos_games, pos_plays = invert(pos_data);
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);
neg_games, neg_plays = invert(neg_data);
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);


idx = 131
pos_dist = (pos_plays[idx] .+ pos_plays[idx + 50] .+ 0.01)./2.03
neg_dist = (neg_plays[idx] .+ neg_plays[idx + 50] .+ 0.01)./2.03

ChisqTest(round.(Int, pos_dist .*30), neg_dist)

i = idx
[((neg_dist[i]*28 - 28*pos_dist[i])^2)/28*pos_dist[i] for i in 1:3] |> sum

χ_vals = map([31, 34, 38, 41, 44, 50, 131, 137, 141, 144, 149]) do idx
    pos_dist = (pos_plays[idx] .+ pos_plays[idx])./2
    neg_dist = (neg_plays[idx] .+ neg_plays[idx])./2
    # sum([(pos_dist[i]*30 - 30*neg_dist[i])^2/30*neg_dist[i] for i in 1:3])
    sum([((neg_dist[i]*28 - 28*pos_dist[i])^2)/28*pos_dist[i] for i in 1:3])
end


χ_vals
sum(χ_vals)

pos_dist = (pos_plays[81] .+ pos_plays[31])./2
neg_dist = (neg_plays[81] .+ neg_plays[31])./2

pos_dist[1]*30
















######################################################
#%% Generate Data Frame For first/last
######################################################
res_dict_fl = deserialize("saved_objects/res_dict")

res_df_fl = DataFrame()
first_last_symbols = [:fit_mh, :opt_mh, :fit_deep, :opt_deep]

comp_idx = comparison_indices(pos_data)
train_idx, test_idx = early_late_indices(pos_data)

function cat_syms(a, b; sep="_")
    Symbol(String(a)*sep*String(b))
end

for (treatment, data) in zip(["Competing", "Common"], [neg_data, pos_data])
    for (data_type, idxs) in zip(["train", "test"], [train_idx, test_idx])
        row_dict = Dict{Any, Any}(:data_type => data_type, :treatment => treatment)
        row_dict[:random] = rand_loss(data[idxs])
        row_dict[:minimum] = min_loss(data[idxs])
        for sym in first_last_symbols
            for (model, treat_data) in zip([res_dict_fl[sym][:neg_model], res_dict_fl[sym][:pos_model]], ["neg", "pos"])
                row_dict[cat_syms(sym,treat_data)] = prediction_loss(model, data, idxs, res_dict_fl[sym][:costs])
            end
        end
        if length(names(res_df_fl)) == 0
            res_df_fl = DataFrame(row_dict)
        else
            push!(res_df_fl, row_dict)
        end
    end
end


CSV.write("results/fl_results_to_plot.csv", res_df_fl)
serialize("saved_objects/res_df_fl",res_df_fl)


######################################################
#%%
######################################################



using Plots
using StatsPlots

pyplot()

keys_to_plot = [:random, :fit_mh_neg, :fit_mh_pos, :opt_mh_neg, :opt_mh_pos, :fit_deep_neg,  :fit_deep_pos, :opt_deep_neg, :opt_deep_pos, :minimum]

plots_vec = []
for data_type in ["train", "test"], treat in ["Competing", "Common"]
    vals = convert(Vector, first(res_df_fl[(res_df_fl.treatment .== treat) .& (res_df_fl.data_type .== data_type), keys_to_plot]))
    ctg = [repeat(["competing"], 4)..., repeat(["commmon"], 4)...]
    nam = [repeat(["fit mh", "opt mh", "fit deep", "opt deep"],2)...]
    bar_vals = hcat(transpose(reshape(vals[2:(end-1)], (2,4))))
    plt = groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*" interest "*data_type*" games", ylims=(0,1.3))
    plot!([vals[1]], linetype=:hline, width=2, label="random loss", color=:grey)
    plot!([vals[2*4 + 2]], linetype=:hline, width=2, label="min loss", color=:black)
    push!(plots_vec, plt)
end

length(plots_vec)
plot(plots_vec..., layout=(2,2), size=(794,1123))

savefig("../overleaf/figs/loglikelihoods_fl.png")

res_df_fl

#= Analysis Plan

1. Implement Train 30 Test 20
2. Implement LOOCV at the population level
3. Fit and optimize MetaHeuritic
    - Fit costs when fitting parameters
4. Fit and optimize Deep Heuristic
5. Fit RuleLearning

=#
