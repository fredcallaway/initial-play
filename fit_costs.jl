using Distributed
addprocs()
@everywhere begin

    using Serialization
    include("model.jl")

    scale(x, low, high) = low + x * (high - low)
    logscale(x, low, high) = exp(scale(x, log(low), log(high)))

    function _transform(x)
        [
            logscale(x[1], 0.01, 0.25),
            logscale(x[2], 0.01, 0.4),
            logscale(x[3], 0.01, 1),
            logscale(x[4], 0.1, 3),
        ]
    end

    all_costs = map(1:1000) do i
        Costs(_transform(rand(4))...)
    end

    function try_costs(costs)
        try
            mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
            train = make_optimize(mh_base, costs)
            model = train(common_data[train_indices])
            test(model, common_data[test_indices])
        catch
            Inf
        end
    end

    uuid() = string(rand(1:100000000), base=62)

    dir = "results/random_search2/"
end # @everywhere

if !isdir(dir)
    mkdir(dir)
end

results = pmap(all_costs) do costs
    res = (costs=costs, loss=try_costs(costs))
    open("$dir/$(uuid())", "w+") do f
        serialize(f, res)
    end
    res
end