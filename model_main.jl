println(1)
println(2)


# %% ====================  ====================
using Distributed
using Serialization

include("model.jl")


using Printf
function evaluate(model, data)
    treatment_idx = setdiff(test_indices, comparison_idx)
    @printf "comparison: %.3f vs. %.3f\n" test(model, data[comparison_idx]) min_loss(data[comparison_idx])
    @printf "treatment:  %.3f vs. %.3f\n" test(model, data[treatment_idx]) min_loss(data[treatment_idx])
    @printf "all:        %.3f vs. %.3f\n" test(model, data) min_loss(data)
end



# %% ====================  ====================
all_costs, losses = open(deserialize, "results/random_search")
xx = map(losses) do x
    isnan(x) ? Inf : x
end

all_costs[argmin(xx)]


# %% ====================  ====================
using Glob
using SplitApplyCombine

results = map(glob("results/random_search2/*")) do f
    open(deserialize, f)
end
costs, losses = invert(results)
losses = map(x->isnan(x) ? Inf : x, losses)

best, idx = findmin(losses)
costs[idx]

# %% ====================  ====================
treatment = "common"
# costs = Costs(0.2, 0.2, 0.1, 1.5)
costs = Costs(0.1682716344671231, 0.18753820682175806, 0.06078936090685586, 1.0690653407809032)
mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(0., 1.), RowMean(2.)])], [0., 0., 0.]);
train = make_optimize(mh_base, costs)

model = train(data[treatment][train_indices])
evaluate(model, data[treatment])

map(comparison_idx[1:cld(length(comparison_idx), 2)]) do idx
    game = common_data[idx][1]
    (round=idx, play=play_distribution(model, game))
end |> DataFrame |> CSV.write("results/$(treatment)_comparison_model.csv")


