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
using Tables





# nprocs() == 1 && addprocs(Sys.CPU_THREADS - 1)
include("Heuristics.jl")  # prevent LoadError: UndefVarError: Game not defined below
@everywhere using ForwardDiff
@everywhere using DataFrames
@everywhere include("Heuristics.jl")
@everywhere include("model.jl")
#%%

all_data = Dict(
    :pos => load_treatment_data("positive"),
    :neg => load_treatment_data("negative"),
)
all_data[:both] = vcat(all_data[:pos], all_data[:neg])

@everywhere all_data = $all_data
train_idx, test_idx = early_late_indices(all_data[:pos])
comp_idx = comparison_indices(all_data[:pos])

both_train_idx, both_test_idx = early_late_indices(all_data[:both])
both_comp_idx = comparison_indices(all_data[:both])

@assert early_late_indices(all_data[:pos]) == early_late_indices(all_data[:neg])
@assert comparison_indices(all_data[:pos]) == comparison_indices(all_data[:neg])

#%%
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


#%%

mh_r = MetaHeuristic([JointMax(1.), RowHeuristic(2., 1.), RowHeuristic(0., 1.), RowHeuristic(-2., 1.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 1.)])], [0., 0., 0., 0., 0.]);
C = Costs(0.349700752494858, 0.28515682342466236, 0.1287109375, 5.5218037538845) # Optim
# C = Costs(0.4944099380364206, 0.18934945036597656, 0.0986328125, 7.185041933540517) # Fit
rl_base = RuleLearning(mh_r, 0.9, 1., C)


# opt_rl = fit_model(rl_base, all_data[:both], both_train_idx, C, n_iter=2)

opt_rl_new = fit_rl(rl_base, all_data[:both], both_train_idx)

opt_β_prior_rl = fit_βs_and_prior(rl_base, all_data[:both], both_train_idx)


# rule_loss(opt_rl_new, all_data[:pos])
# rule_loss(opt_rl_new, all_data[:neg])
#
# opt_rl.β₀
# opt_rl.β₁

rule_loss_idx(opt_rl_new, all_data[:neg], test_idx)
rule_loss_idx(opt_rl_new, all_data[:pos], test_idx)

rule_loss_idx(opt_β_prior_rl, all_data[:neg], test_idx)
rule_loss_idx(opt_β_prior_rl, all_data[:pos], test_idx)


dat =  all_data[:both]
games, plays = invert(dat)
empirical_play = CacheHeuristic(games, plays);



mh = MetaHeuristic([JointMax(1.), RowHeuristic(-2., 1.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 1.)])], [0., 0., 0.]);

opt_model_no_RI = fit_model_no_RI(mh, all_data[:neg][train_idx])

c_fit = Costs(0.4944099380364206, 0.18934945036597656, 0.0986328125, 7.185041933540517) # Fit
opt_model = fit_model(mh, all_data[:neg][train_idx], c_fit)

prediction_loss_no_RI(opt_model_no_RI, all_data[:neg], test_idx, c_fit)
prediction_loss(opt_model, all_data[:neg], test_idx, c_fit)

pos_opt_model_no_RI = fit_model_no_RI(mh, all_data[:pos][train_idx])
pos_opt_model = fit_model(mh, all_data[:pos][train_idx], c_fit)
prediction_loss_no_RI(pos_opt_model_no_RI, all_data[:pos], test_idx, c_fit)
prediction_loss(pos_opt_model, all_data[:pos], test_idx, c_fit)

fin_rules = end_rules(opt_rl, all_data[:neg])
fin_rules[1]["col"]

my_softmax(fin_rules[1]["col"].prior)

opt_mh = fit_model(mh_r, all_data[:pos], test_idx, C)
prediction_loss(opt_mh, all_data[:pos], test_idx, C)

opt_mh
neg_opt_mh
