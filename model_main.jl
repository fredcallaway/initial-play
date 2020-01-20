using Flux
using JSON
using CSV
using DataFrames
# using DataFramesMeta
using SplitApplyCombine
using Random
using Glob
using Distributed
# using BSON
using Serialization
using StatsBase
using Statistics
using Sobol

nprocs() == 1 && addprocs(Sys.CPU_THREADS - 1)
include("Heuristics.jl")  # prevent LoadError: UndefVarError: Game not defined below
@everywhere begin
    include("Heuristics.jl")
    include("rule_learning.jl")
    include("model.jl")
    include("box.jl")
end

include("gp_min.jl")

# %% ==================== Load Data ====================
all_data = Dict(
    :pos => load_treatment_data("positive"),
    :neg => load_treatment_data("negative"),
)
@everywhere all_data = $all_data
train_idx, test_idx = early_late_indices(all_data[:pos])
comp_idx = comparison_indices(all_data[:pos])

@assert early_late_indices(all_data[:pos]) == early_late_indices(all_data[:neg])
@assert comparison_indices(all_data[:pos]) == comparison_indices(all_data[:neg])
# %% ==================== Fitting costs ====================

DEBUG = true
DEBUG && @warn "DEBUG MODE: not computing loss"


<<<<<<< HEAD
get_cost_type(model) = Costs
get_cost_type(model::Chain) = DeepCosts
=======
>>>>>>> 6d277fc4ba5c23554a0e2e20e0273abcdf5481c7
@everywhere begin
    get_cost_type(model) = Costs
    get_cost_type(model::Chain) = DeepCosts
end
<<<<<<< HEAD


=======
>>>>>>> 6d277fc4ba5c23554a0e2e20e0273abcdf5481c7

function make_loss(model, train::Function, space::Box; parallel=true)
    mymap = parallel ? pmap : map

    function loss(x)
        DEBUG && return sum(x)
        costs = get_cost_type(model)(;space(x)...)
        mymap(values(all_data)) do data
            trained_model = train(model, data, train_idx, costs)
            prediction_loss(trained_model, data, train_idx, costs)
        end |> sum
    end
end

function init_points(dim, n);
    seq = SobolSeq(dim)
    skip(seq, n)
    [next!(seq) for i in 1:n]
end

<<<<<<< HEAD
# function fit_costs(model, train, space; n_sobol=500, n_gp=50)
function fit_costs(model, train, space; n_sobol=64, n_gp=5)
    f = make_loss(model, optimize_model, space; parallel=false)
=======
function fit_costs(model, train, space; n_sobol=500, n_gp=50)
    println("Training $(typeof(model).name) model with $train")
    f = make_loss(model, train, space; parallel=false)
>>>>>>> 6d277fc4ba5c23554a0e2e20e0273abcdf5481c7
    xs = init_points(n_free(space), n_sobol)
    @time ys = pmap(f, xs)
    f = make_loss(model, train, space; parallel=true)
    @time gp_opt = gp_minimize(f, n_free(space), init_Xy=(combinedims(xs), ys), iterations=n_gp)

    costs = get_cost_type(model)(;space(gp_opt.observed_optimizer)...)
    trained_models = pmap(collect(all_data)) do (treat, data)
        treat => train(model, data, train_idx, costs)
    end |> Dict

    costs, trained_models, gp_opt
end

# %% ==================== MetaHeuristic ====================
mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
mh_space = Box(
    :α => (0.05, 0.5, :log),
    :λ => (0.05, 0.3, :log),
    :level => (0., 0.2),
    :m_λ => (0.5,2.5, :log),
)

mh_results = Dict(
    :fit => fit_costs(mh_base, fit_model, mh_space),
    :opt => fit_costs(mh_base, optimize_model, mh_space)
)

# %% save results

function results_df(results)
    df = mapmany(collect(results)) do (mode, res)
        costs, trained_models = res
        mapmany(trained_models) do (train_treat, model)
            map([:neg, :pos]) do test_treat
                y = prediction_loss(model, all_data[test_treat], test_idx, costs)
                # println("$mode $treat $(round(y; digits=3))",)
                (test=test_treat, mode=mode, train=train_treat, loss=y)
            end
        end
    end |> DataFrame
    sort!(df, (:test, :mode))
end

result_df(results) |> CSV.write("results/gp_mh.csv")


map(collect(keys(all_data))) do treat
    d = all_data[treat][test_idx]
    (treat=treat, rand=rand_loss(d), min=min_loss(d))
end |> CSV.write("results/rand_min.csv")

# %% ====================  ====================


# %% Setup and run
qch_base = QCH(0.3, 0.3, 1.)
cs = Cost_Space((0.05, 0.3), (0.05, 0.3), (0., 0.2), (0.5,2.5))


# %% ====================  ====================
# deep_base = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))
deep_base = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense(50,50), Game_Soft(50), Action_Response(1), Last(2));

deep_space = Box(
    :γ => (0.01,0.1),
    :exact => (0.001, 0.1, :log),
    :sim => (0.0, 0.3),
)

deep_results = Dict(
    :fit => fit_costs(deep_base, fit_model, deep_space),
    :opt => fit_costs(deep_base, optimize_model, deep_space)
)

results_df(deep_results) |> CSV.write("results/deep.csv")

# %% ====================  ====================


# rl_base = RuleLearning(deepcopy(mh_base), 1., 1., rand(cs))


#%% Run
n_runs = 4*64
n_runs_deep = 64
mh_costs_vec  = rand(cs, n_runs)
deep_costs_vec  = rand(deep_cs, n_runs_deep)


res_dict = Dict()
save_file = "saved_objects/res_dict5"
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

#%%
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
res_dict = deserialize("saved_objects/res_dict5")



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

######################################################
#%% Generate Data Frame For First Last Train Test data
######################################################
res_dict = deserialize("saved_objects/res_dict5")



symbols = [:fit_mh, :opt_mh, :fit_deep, :opt_deep]

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
    train_idx, test_idx = early_late_indices(pos_data)
    for sym in symbols
        for (model, treat_data) in zip([res_dict[sym][:neg_model], res_dict[sym][:pos_model]], ["neg", "pos"])
            train_dict[cat_syms(sym,treat_data)] = prediction_loss(model, data, train_idx, res_dict[sym][:costs])
            test_dict[cat_syms(sym,treat_data)] = prediction_loss(model, data, test_idx, res_dict[sym][:costs])
        end
    end

    if length(names(res_df)) == 0
        res_df = DataFrame(Dict(k=>typeof(v)[] for (k,v) in train_dict))
    end
    push!(res_df, train_dict)
    push!(res_df, test_dict)
end


res_df

CSV.write("results/FirstLast_results_to_plot.csv", res_df)
serialize("saved_objects/res_df_FL",res_df)
#%% Plot
res_df = deserialize("saved_objects/res_df_FL")

using Plots
using StatsPlots
# ENV["PYTHON"]=""
# Pkg.build("PyCall")
pyplot()

keys_to_plot = [:random, :fit_mh_neg, :fit_mh_pos, :opt_mh_neg, :opt_mh_pos, :fit_deep_neg,  :fit_deep_pos, :opt_deep_neg, :opt_deep_pos, :minimum]
# keys_to_plot = [:random, :fit_mh_cv_neg_mean, :fit_mh_cv_pos_mean, :opt_mh_cv_neg_mean, :opt_mh_cv_pos_mean, :fit_deep_cv_neg_mean,  :fit_deep_cv_pos_mean, :opt_deep_cv_neg_mean, :opt_deep_cv_pos_mean, :minimum]

plots_vec = []
for data_type in ["train", "test"], treat in ["Competing", "Common"]
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
# full_data_res_dict_load = deepcopy(res_dict)
full_data_res_dict = deepcopy(res_dict)


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
# for fit_type in [:Random, :Maximum, :Actual]
#     for treat in [:neg, :pos]
        for (data_name, data) in zip([:Common, :Competing], [pos_data, neg_data])
            # m_sym = cat_syms(treat, :model)
            # m = full_data_res_dict[fit_type][m_sym]
            # costs = full_data_res_dict[fit_type][:costs]
            games, plays = invert(data);
            empirical_play = CacheHeuristic(games, plays);
            max_dict[data_name] = mean([max_payoff(empirical_play, g) for g in games])
            rand_dict[data_name] = mean([rand_payoff(empirical_play, g) for g in games])
            actual_dict[data_name] = mean([expected_payoff(empirical_play, empirical_play, g) for g in games])
        end
#     end
# end
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
               vec(gen_row(:Random)), # ragged!
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

idx = 49
indicies = [idx + 50*i for i in 0:11]

pos_dist = sum(pos_plays[indicies])/length(indicies)
neg_dist = sum(neg_plays[indicies])/length(indicies)

# pos_dist = (pos_plays[idx] .+ pos_plays[idx + 50] .+ 0.01)./2.03
# neg_dist = (neg_plays[idx] .+ neg_plays[idx + 50] .+ 0.01)./2.03

ChisqTest(round.(Int, neg_dist .*180), pos_dist)
ChisqTest(round.(Int, pos_dist .*180), neg_dist)

pos_dist'*pos_games[idx].row*pos_dist
neg_dist'*neg_games[idx].row*neg_dist



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





index = 31
expected_payoff(pos_empirical_play, pos_empirical_play, pos_games[index])
expected_payoff(neg_empirical_play, neg_empirical_play, neg_games[index])










######################################################
#%% Generate Data Frame For first/last
######################################################
res_dict_fl = deserialize("saved_objects/res_dict5")

res_df_fl = DataFrame()
first_last_symbols = [:fit_mh, :opt_mh, :fit_deep, :opt_deep]
res_df_fl
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
#%% Simple compare of payoffs for model estimated on first
data_mod = :opt_mh
fit_mh_neg = res_dict_fl[data_mod][:neg_model]
fit_mh_pos = res_dict_fl[data_mod][:pos_model]

costs = full_data_res_dict[data_mod][:costs]

no_comparison_idx = setdiff(1:200, comp_idx)

games, plays = invert(neg_data[no_comparison_idx]);
empirical_play = CacheHeuristic(games, plays);
mean([expected_payoff(fit_mh_neg, empirical_play, g, costs) for g in games])
mean([expected_payoff(fit_mh_pos, empirical_play, g, costs) for g in games])
mean([perf(fit_mh_pos, [g], empirical_play, costs) for g in games])
mean([perf(fit_mh_neg, [g], empirical_play, costs) for g in games])





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
plot(plots_vec..., layout=(2,2), size=(1000,1000))

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

#%% Generate combined data for heatmap and marginal distributions plotting
pos_df = DataFrame()
neg_df = DataFrame()
pos_games, pos_plays = invert(pos_data);
neg_games, neg_plays = invert(neg_data);
for i in [31, 38, 42, 49]
    row_idx = filter(x -> x%100 == i, 1:length(pos_data))
    col_idx = filter(x -> (x-50)%100 == i, 1:length(pos_data))
    pos_row_play = mean([pos_plays[i] for i in row_idx])
    pos_col_play = mean([pos_plays[i] for i in col_idx])
    neg_row_play = mean([neg_plays[i] for i in row_idx])
    neg_col_play = mean([neg_plays[i] for i in col_idx])
    g = pos_games[i]
    g = [[Int(round(g.row[j,i])), Int(round(g.col[j,i]))] for i in 1:3, j in 1:3]
    pos_dict = Dict(:round => i, :row_play => JSON.json(Tuple(pos_row_play)), :col_play => JSON.json(Tuple(pos_col_play)), :row_game => JSON.json(g), :col_game => JSON.json(g), :treatment => "positive", :type => "comparison")
    neg_dict = Dict(:round => i, :row_play => JSON.json(Tuple(neg_row_play)), :col_play => JSON.json(Tuple(neg_col_play)), :row_game => JSON.json(g), :col_game => JSON.json(g), :treatment => "negative", :type => "comparison")
    if length(names(pos_df)) == 0
        pos_df = DataFrame(pos_dict)
        neg_df = DataFrame(neg_dict)
    else
        push!(pos_df, pos_dict)
        push!(neg_df, neg_dict)
    end
end

CSV.write("results/pos_to_plot.csv", pos_df)
CSV.write("results/neg_to_plot.csv", neg_df)
