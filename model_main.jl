cd("/usr/people/flc2/juke/initial-play")

using Distributed
addprocs(48)
@everywhere include("model.jl")
model = make_fit(QCH())(games)
@time res = cross_validate(make_fit(QCH()), test, games; k=4)

# %% ====================  ====================
mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);

costs = Costs(0.1, 0.1, 0.2, 1.5)
make_optimize(mh_base, costs)(games)

costs = Costs(0.1, 0.1, 0.2, 2.5)
make_optimize(mh_base, costs)(games)


# %% ====================  ====================
xs = 0.1:0.1:4
all_costs = [Costs(0.1, 0.1, 0.2, x) for x in xs]
@time res = pmap(all_costs) do costs
    # make_optimize(mh_base, costs)(games)
    mean(cross_validate(make_optimize(mh_base, costs), test, games; k=4))
end

plot(xs, res, label="")
xlabel!("Rational Inattention Cost")
ylabel!("Loss")
savefig("ri_cost.pdf")
# %% ====================  ====================

ranges = [
    (1:40) / 100,
    (1:40) / 100,
    (1:40) / 100,
    (1:40) / 10,
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
        mean(cross_validate(make_optimize(mh_base, costs), test, games; k=10))
    catch
        missing
    end
end

x = map(results) do r
    ismissing(r) || isnan(r) ? Inf : r
end
# %% ====================  ====================
xs = 0.01:0.01:0.4
all_costs = [Costs(0.1, 0.1, x, 0.8) for x in xs]
@time res = pmap(all_costs) do costs
    # make_optimize(mh_base, costs)(games)
    mean(cross_validate(make_optimize(mh_base, costs), test, games; k=4))
end

plot(xs, res, label="")
xlabel!("Level Cost")
ylabel!("Loss")
savefig("ri_cost.pdf")
