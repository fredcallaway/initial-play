include("Heuristics.jl")

function optimize(model, costs)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    for i in 1:n_iter
        optimize_h!(model, games, empirical_play, costs)
        opt_prior!(model, games, empirical_play, costs)
    end
    model
end