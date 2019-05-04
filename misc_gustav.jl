using Serialization
using Pkg
include("Heuristics.jl")

res = open(deserialize, "results/cost_cross")

print(res)

for i in 1:length(res[1])
    println(res[1][i], res[2][i])
end

non_missing_costs = res[1][res[2] .!== missing]
non_missing_loss = res[2][res[2] .!== missing]

clean_costs = non_missing_costs[.!(isnan.(non_missing_loss))]
clean_loss = non_missing_loss[.!(isnan.(non_missing_loss))]

min_idx = argmin(clean_loss)

non_missing_costs[min_idx]

clean_costs[106:end]
clean_loss[106:end]
