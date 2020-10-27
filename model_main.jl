using Flux
using JSON
using CSV
using DataFrames
using DataFramesMeta
using SplitApplyCombine
using Random
using Glob
using Distributed
# using BSON
using Serialization
using StatsBase
using Statistics
using Sobol
using Profile



nprocs() == 1 && addprocs(Sys.CPU_THREADS - 1)
include("Heuristics.jl")  # prevent LoadError: UndefVarError: Game not defined below
@everywhere begin
    include("Heuristics.jl")
    include("model.jl")
    include("box.jl")
end

include("gp_min.jl")



n_sobol = 600

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

DEBUG = false
DEBUG && @warn "DEBUG MODE: not computing loss"

@everywhere begin
    get_cost_type(model) = Costs
    get_cost_type(model::Chain) = DeepCosts
end

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

# This one is without BayesianOptimization
# function fit_costs(model, train, space; n_sobol=n_sobol)
#     f = make_loss(model, train, space; parallel=false)
#     xs = init_points(n_free(space), n_sobol)
#     println("Fitting $model via $train and ", n_sobol, " Sobol points")
#     @time ys = pmap(f, xs)
#     min_idx = argmin(ys)
#     costs = get_cost_type(model)(;space(xs[min_idx])...)
#     trained_models = pmap(collect(all_data)) do (treat, data)
#         treat => train(model, data, train_idx, costs)
#     end |> Dict
#
#     costs, trained_models
# end

function results_df(results; test_idx=test_idx)
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

#### Uncomment below for BayesianOptimization
function fit_costs(model, train, space; n_sobol=64, n_gp=5)
    f = make_loss(model, train, space; parallel=false)
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
    :fit => fit_costs(mh_base, fit_model, mh_space; n_sobol=n_sobol, n_gp=20),
    :opt => fit_costs(mh_base, optimize_model, mh_space; n_sobol=n_sobol, n_gp=20)
)
# %% save results

function prediction_loss(results, key, treat, data, idx)
    m = results[key][2][treat]
    costs = results[key][1]
    prediction_loss(m, data, idx, costs)
end



serialize("saved_objects/mh_results_"*string(n_sobol), mh_results)
res_df_mh = results_df(mh_results)
res_df_mh |> CSV.write("results/gp_mh_"*string(n_sobol)*".csv")

results_df(mh_results; test_idx=comp_idx) |> CSV.write("results/mh_"*string(n_sobol)*"comp.csv")

map(collect(keys(all_data))) do treat
    d = all_data[treat][test_idx]
    (treat=treat, rand=rand_loss(d), min=min_loss(d))
end |> CSV.write("results/rand_min.csv")

map(collect(keys(all_data))) do treat
    d = all_data[treat][comp_idx]
    (treat=treat, rand=rand_loss(d), min=min_loss(d))
end |> CSV.write("results/rand_min_comp.csv")


mh_results[:opt][1]

# %% ==================== Noisy Best Reply  ====================
noisy_base = FSBR(1., 0.5, 0.5)

function train_NosiyBR(dat)
    noisy_base = FSBR(1., 0.5, 0.5)
    games, plays = invert(dat)
    empirical_play = CacheHeuristic(games, plays);
    trained_model = fit_h!(noisy_base, games, empirical_play, empirical_play, nothing)
    trained_model
end

pos_trained_model = train_NosiyBR(all_data[:pos][train_idx])
neg_trained_model = train_NosiyBR(all_data[:neg][train_idx])

pos_test_dat =  all_data[:pos][test_idx]
pos_games, pos_plays = invert(pos_test_dat)
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);

neg_test_dat =  all_data[:neg][test_idx]
neg_games, neg_plays = invert(neg_test_dat)
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);

prediction_loss(pos_trained_model, pos_games, pos_empirical_play, pos_empirical_play)
prediction_loss(neg_trained_model, pos_games, pos_empirical_play, pos_empirical_play)
prediction_loss(pos_trained_model, neg_games, neg_empirical_play, neg_empirical_play)
prediction_loss(neg_trained_model, neg_games, neg_empirical_play, neg_empirical_play)


# %% ==================== GCH ====================
gch_base = GCH(1., 0.5, 0.5)

function train_GCH(dat)
    gch_base = GCH(1., 0.5, 0.5)
    games, plays = invert(dat)
    empirical_play = CacheHeuristic(games, plays);
    trained_model = fit_h!(gch_base, games, empirical_play)
    trained_model
end

pos_trained_model = train_GCH(all_data[:pos][train_idx])
neg_trained_model = train_GCH(all_data[:neg][train_idx])

prediction_loss(GCH(2.1, 2.1, 1.1), pos_games, pos_empirical_play)

pos_test_dat =  all_data[:pos][test_idx]
pos_games, pos_plays = invert(pos_test_dat)
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);

neg_test_dat =  all_data[:neg][test_idx]
neg_games, neg_plays = invert(neg_test_dat)
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);

prediction_loss(pos_trained_model, pos_games, pos_empirical_play)
prediction_loss(neg_trained_model, pos_games, pos_empirical_play)
prediction_loss(pos_trained_model, neg_games, neg_empirical_play)
prediction_loss(neg_trained_model, neg_games, neg_empirical_play)


# %% ====================  ====================
# %% Setup and run
qch_base = QCH(0.3, 0.3, 1.)
qch_space = Box(
    :α => (0.05, 0.5, :log),
    :λ => (0.05, 0.3, :log),
    :level => (0., 0.2),
    :m_λ => (0.5,2.5, :log),
)

@time qch_results = Dict(
    :fit => fit_costs(qch_base, fit_model, qch_space; n_sobol=128),
    :opt => fit_costs(qch_base, optimize_model, qch_space; n_sobol=128)
)

res_df_qch = results_df(qch_results)

serialize("saved_objects/qch_results", qch_results)
results_df(qch_results) |> CSV.write("results/qch.csv")

results_df(qch_results; test_idx=comp_idx) |> CSV.write("results/qch_comp.csv")


# %% ====================  ====================
# deep_base = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))
deep_base = Chain(Game_Dense(15, 50), Game_Soft(50), Action_Response(1), Last(2));

deep_space = Box(
    :γ => (0.01,0.1),
    :exact => (0.001, 0.3, :log),
    :sim => (0.0, 0.3),
)

@time deep_results = Dict(
    :fit => fit_costs(deep_base, fit_model, deep_space; n_sobol=n_sobol),
    :opt => fit_costs(deep_base, optimize_model, deep_space; n_sobol=n_sobol)
)

res_df_deep = results_df(deep_results)

serialize("saved_objects/deep_results_"*string(n_sobol), deep_results)
results_df(deep_results) |> CSV.write("results/deep_"*string(n_sobol)*".csv")

results_df(deep_results; test_idx=comp_idx) |> CSV.write("results/deep_"*string(n_sobol)*"comp.csv")


res_mh_512 = CSV.read("results/gp_mh_512.csv")
res_mh_576 = CSV.read("results/gp_mh_576.csv")


res_mh_512[:loss] .- res_mh_576.loss
# Generate optimal behavior without cognitive costs to approximate actual optimal behavior

no_costs = DeepCosts(0.0, 0.0, 0.0)
no_costs_opt = Dict(:neg => optimize_model(deep_base, all_data[:neg], train_idx, no_costs), :pos => optimize_model(deep_base, all_data[:pos], train_idx, no_costs))

neg_opt_payoffs = Dict(:Common => mean_payoff(no_costs_opt[:neg], all_data[:pos][test_idx]), :Competing => mean_payoff(no_costs_opt[:neg], all_data[:neg][test_idx]))
pos_opt_payoffs = Dict(:Common => mean_payoff(no_costs_opt[:pos], all_data[:pos][test_idx]), :Competing => mean_payoff(no_costs_opt[:pos], all_data[:neg][test_idx]))

# %% ====================  ====================

######################################################
#%% Generate payoff for different heuristics DF
######################################################

# deep_results = deserialize("saved_objects/deep_results")
deep_results = deserialize("saved_objects/deep_results_512")
deserialize("saved_objects/deep_results")
deserialize("saved_objects/deep_results_512")
deserialize("saved_objects/deep_results_576")
# mh_results = deserialize("saved_objects/mh_results")
mh_results = deserialize("saved_objects/mh_results_512")
deserialize("saved_objects/mh_results")
deserialize("saved_objects/mh_results_512")
deserialize("saved_objects/mh_results_576")
deserialize("saved_objects/mh_results_600")

results_df(deserialize("saved_objects/deep_results_576"))
results_df(deserialize("saved_objects/mh_results_600"))
results_df(deserialize("saved_objects/qch_results"))

function mean_payoff(h::Chain, d::Data)
    games, play = invert(d)
    opp = CacheHeuristic(games, play);
    mean([expected_payoff(h, opp, g) for g in games])
end


function expected_payoff(h::Chain, opponent::Heuristic, g::Game)
    p = h(g)
    p_opp = play_distribution(opponent, transpose(g))
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
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

pos_data = all_data[:pos][test_idx]
neg_data = all_data[:neg][test_idx]
pos_games, pos_plays = invert(pos_data)
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);
neg_games, neg_plays = invert(neg_data)
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);


payoff_df = DataFrame()

for (results, res_names) in zip([mh_results, deep_results], [:mh, :deep])
    for estimation in [:fit, :opt]
        for treat in [:neg, :pos]
            row_dict = Dict()
            for (data_name, data) in zip([:Common, :Competing], [pos_data, neg_data])
                m = results[estimation][2][treat]
                costs = results[estimation][1]
                games, plays = invert(data);
                empirical_play = CacheHeuristic(games, plays);
                row_dict[:estim_treat] = treat
                row_dict[:estimation] =  estimation
                row_dict[:model] = res_names
                row_dict[data_name] = mean([expected_payoff(m, empirical_play, g, costs) for g in games])
            end
            if length(names(payoff_df)) == 0
                payoff_df = DataFrame(row_dict)
            else
                push!(payoff_df, row_dict)
            end
        end
    end
end
payoff_df
rand_dict = Dict{Symbol, Any}(:model => :Random, :estimation => Symbol(""), :estim_treat => Symbol(""))
max_dict = Dict{Symbol, Any}(:model => :Maximum, :estimation => Symbol(""), :estim_treat => Symbol(""))
actual_dict = Dict{Symbol, Any}(:model => :Actual, :estimation => Symbol(""), :estim_treat => Symbol(""))

no_costs_deep_pos = Dict{Symbol, Any}(:model => :deep_no_cost, :estimation => :opt, :estim_treat => :pos)
no_costs_deep_neg = Dict{Symbol, Any}(:model => :deep_no_cost, :estimation => :opt, :estim_treat => :neg)


for (data_name, data) in zip([:Common, :Competing], [pos_data, neg_data])
    games, plays = invert(data);
    empirical_play = CacheHeuristic(games, plays);
    max_dict[data_name] = mean([max_payoff(empirical_play, g) for g in games])
    rand_dict[data_name] = mean([rand_payoff(empirical_play, g) for g in games])
    actual_dict[data_name] = mean([expected_payoff(empirical_play, empirical_play, g) for g in games])
    no_costs_deep_neg[data_name] = mean([expected_payoff(no_costs_opt[:neg], empirical_play, g) for g in games])
    no_costs_deep_pos[data_name] = mean([expected_payoff(no_costs_opt[:pos], empirical_play, g) for g in games])
end

push!(payoff_df, no_costs_deep_pos)
push!(payoff_df, no_costs_deep_neg)
push!(payoff_df, rand_dict)
push!(payoff_df, actual_dict)
push!(payoff_df, max_dict)

payoff_df = payoff_df[:, [:model, :estimation, :estim_treat, :Common, :Competing]]

# Print table
using LaTeXTabulars
using LaTeXStrings


# rename_dict = Dict(:estimation => :Estimation, :estim_treat => Symbol("Estimation data"), :model => :Model)

# rename!(payoff_df, rename_dict)

payoff_df[:model] = string.(payoff_df[:model])
payoff_df[:model] = replace.(payoff_df[:model], "mh" => "Meta heuristic")
payoff_df[:model] = replace.(payoff_df[:model], "deep" => "Deep heuristic")
payoff_df[:model] = replace.(payoff_df[:model], "deep_no_cost" => "Deep with zero cost")

payoff_df[:estimation] = string.(payoff_df[:estimation])
payoff_df[:estimation] = replace.(payoff_df[:estimation], "fit" => "Fit")
payoff_df[:estimation] = replace.(payoff_df[:estimation], "opt" => "Optimize")
payoff_df[:estim_treat] = string.(payoff_df[:estim_treat])
payoff_df[:estim_treat] = replace.(payoff_df[:estim_treat], "neg" => "Competing")
payoff_df[:estim_treat] = replace.(payoff_df[:estim_treat], "pos" => "Commmon")
payoff_df[:Common] = round.(payoff_df[:Common], digits=2)
payoff_df[:Competing] = round.(payoff_df[:Competing], digits=2)
payoff_df
@where(payoff_df, :model .== "Meta heuristic")

latex_tabular("./../overleaf/tex_snippets/payoff_mh.tex",
              Tabular("lllcc"),
              [
               ["Model", "Estimation", "Estimation data", "Payoff in Common", "Payoff in Competing"],
               Rule(:mid),
               # Matrix(payoff_df[1:end-1,:]),
               Matrix(@where(payoff_df, :model .== "Meta heuristic")),
               Rule(),
               Matrix(payoff_df[end-2:end,:]),
               Rule(:bottom)])

latex_tabular("./../overleaf/tex_snippets/payoff_deep.tex",
              Tabular("lllcc"),
              [["Model", "Estimation", "Estimation data", "Payoff in Common", "Payoff in Competing"],
               Rule(:mid),
               # Matrix(payoff_df[1:end-1,:]),
               Matrix(@where(payoff_df, :model .== "Deep heuristic")),
               Rule(),
               Matrix(payoff_df[end-2:end,:]),
               Rule(:bottom)])


######################################################
# %% Tables for estimated heuristics
######################################################
fit_mh_pos = mh_results[:fit][2][:pos]
fit_mh_neg = mh_results[:fit][2][:neg]
opt_mh_pos = mh_results[:opt][2][:pos]
opt_mh_neg = mh_results[:opt][2][:neg]


costs_fit = mh_results[:fit][1]
costs_opt = mh_results[:opt][1]

pos_games, pos_plays = invert(pos_data);
pos_empirical_play = CacheHeuristic(pos_games, pos_plays);
neg_games, neg_plays = invert(neg_data);
neg_empirical_play = CacheHeuristic(neg_games, neg_plays);




opt_h_dists_pos = [h_distribution(opt_mh_pos, g, pos_empirical_play, costs_opt) for g in pos_games];
avg_opt_h_dist_pos = mean(opt_h_dists_pos)
opt_h_dists_neg = [h_distribution(opt_mh_neg, g, neg_empirical_play, costs_opt) for g in neg_games];
avg_opt_h_dist_neg = mean(opt_h_dists_neg)
fit_h_dists_pos = [h_distribution(fit_mh_pos, g, pos_empirical_play, costs_fit) for g in pos_games];
avg_fit_h_dist_pos = mean(fit_h_dists_pos)
fit_h_dists_neg = [h_distribution(fit_mh_neg, g, neg_empirical_play, costs_fit) for g in neg_games];
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






######################################################
#%% Generate combined data for heatmap and marginal distributions plotting
####################################################
pos_df = DataFrame()
neg_df = DataFrame()
pos_games, pos_plays = invert(all_data[:pos]);
neg_games, neg_plays = invert(all_data[:neg]);
for i in [31, 38, 42, 49]
    row_idx = filter(x -> x%100 == i, 1:length(all_data[:pos]))
    col_idx = filter(x -> (x-50)%100 == i, 1:length(all_data[:pos]))
    pos_row_play = mean([pos_plays[i] for i in row_idx])
    pos_col_play = mean([pos_plays[i] for i in col_idx])
    neg_row_play = mean([neg_plays[i] for i in row_idx])
    neg_col_play = mean([neg_plays[i] for i in col_idx])
    g = pos_games[i]
    g = [[Int(round(g.row[i,j])), Int(round(g.col[i,j]))] for i in 1:3, j in 1:3]
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


######################################################
#%% Look at behavior in some games
####################################################
treat = :neg

games, play = invert(all_data[treat])
empirical_play = CacheHeuristic(games, plays);


mh_fit = mh_results[:fit][2][treat]
mh_fit_costs = mh_results[:fit][1]
mh_opt = mh_results[:opt][2][treat]
mh_opt_costs = mh_results[:opt][1]

deep_fit = deep_results[:fit][2][treat]
deep_fit_costs = deep_results[:fit][1]
deep_opt = deep_results[:opt][2][treat]
deep_opt_costs = deep_results[:opt][1]

games, plays = invert(all_data[treat])
empirical_play = CacheHeuristic(games, plays)


function print_game(idx)
    g = games[idx]
    g_mat = [string(Int(round(g.row[i,j])))*","*string(Int(round(g.col[i,j]))) for i in 1:3, j in 1:3]
    print_df = DataFrame()
    print_df[!,:mh_fit] = round.(play_distribution(mh_fit, g, empirical_play, mh_fit_costs), digits=2)
    print_df[!,:mh_opt] = round.(play_distribution(mh_opt, g, empirical_play, mh_opt_costs), digits=2)
    print_df[!,:deep_fit] = round.(deep_fit(g), digits=2)
    print_df[!,:deep_opt] = round.(deep_opt(g), digits=2)

    p_opp = round.(play_distribution(empirical_play, transpose(g)), digits=2)
    print_df[!,:actual] = round.(deepcopy(plays[idx]), digits=2)
    print_df[!,:payoff] = round.(sum(g.row .* p_opp', dims=2)[:], digits=2)


    print_df[!,:c1] = g_mat[:,1]
    print_df[!,:c2] = g_mat[:,2]
    print_df[!,:c3] = g_mat[:,3]
    opp_play_dict = Dict{Symbol, Any}(key => 0 for key in names(print_df)[1:end-2])
    opp_play_dict[:c1] = string(p_opp[1])
    opp_play_dict[:c2] = string(p_opp[2])
    opp_play_dict[:c3] = string(p_opp[3])
    push!(print_df, opp_play_dict)
    display(print_df)
    print_df
end

print_game(129)
