using Optim
include("PHeuristics.jl")

function opt_h_grad(opp_h::Union{Heuristic,Vector{Vector{Float64}}}, games)
    if isa(opp_h, Heuristic)
        opp_plays = opp_cols_played(opp_h, games)
    else
        opp_plays = opp_h
    end

    function opt_fun(params)
        println(params)
        h = Heuristic(params...)
        return -payoff(h, opp_plays, games)
    end

    initial_x = [0., 0., 1.]
    res = Optim.minimizer(optimize(opt_fun, initial_x, BFGS(); autodiff = :forward))
    println(res)
    res
    # for (name, param) in zip(param_names, best_candidate(res))
    #     setfield!(h, name, param)
    # end
end
n_games = 1000
game_size = 3
ρ = 0.
training_games = [Game(game_size, ρ) for i in range(1,n_games)];
res = opt_h_grad(Heuristic(0., 0., 10.), training_games)




# =============================================
