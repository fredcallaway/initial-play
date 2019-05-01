using Distributed
using Serialization
addprocs(96)
@everywhere begin

    include("model.jl")

    scale(x, low, high) = low + x * (high - low)
    logscale(x, low, high) = exp(scale(x, log(low), log(high)))

    function transform(x)
        [
            logscale(x[1], 0.01, 0.25),
            logscale(x[2], 0.01, 0.4),
            logscale(x[3], 0.01, 1),
            logscale(x[4], 0.1, 3),
        ]
    end

    all_costs = map(1:1000) do i
        Costs(transform(rand(4))...)
    end

    function try_costs(costs)
        try
            mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
            common_logp = cross_validate(make_optimize(mh_base, costs), test, common_data; k=10)
            competing_logp = cross_validate(make_optimize(mh_base, costs), test, competing_data; k=10)
            mean([common_logp; competing_logp])
        catch
            Inf
        end
    end

end # @everywhere

results = pmap(try_costs, all_costs)

open("results/random_search", "w+") do f
    serialize(f, (
        all_costs=all_costs,
        loss=results
    ))
end


all_costs, losses = open(deserialize, "results/random_search")
xx = map(losses) do x
    isnan(x) ? Inf : x
end

all_costs[argmin(xx)]

try_costs(Costs(0.1, 0.1, 0.2, 0.8))