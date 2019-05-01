using Distributed
addprocs(48)
@everywhere include("model.jl")

# %% ====================  ====================
mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
costs = Costs(0.1, 0.1, 0.2, 0.8)

ranges = [
    (5:40) / 100,
    (5:40) / 100,
    (5:40) / 100,
    (5:40) / 10,
]
fields = fieldnames(Costs)
base_cost = Costs(0.1, 0.1, 0.2, 0.8)

function setfield(x, k::Symbol, v)
    x = deepcopy(x)
    setfield!(x, k, v)
    x
end

all_costs = Costs[]
for i in 1:4
    for x in ranges[i]
        push!(all_costs, setfield(base_cost, fields[i], x))
    end
end

# %% ====================  ====================
results = pmap(all_costs) do costs
    # make_optimize(mh_base, costs)(games)
    try
        common_logp = cross_validate(make_optimize(mh_base, costs), test, common_data; k=10)
        competing_logp = cross_validate(make_optimize(mh_base, costs), test, competing_data; k=10)
        mean([common_logp; competing_logp])
    catch
        missing
    end
end