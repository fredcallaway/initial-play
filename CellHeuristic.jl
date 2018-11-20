include("PHeuristics.jl")

mutable struct CellHeuristic
    α::Real
    λ::Real
end

#%%

function normalize(g::Game)
    Game(g.row .- mean(g.row), g.col .- mean(g.col))
end

function cell_values(h::CellHeuristic, game::Game)
    g = normalize(game)
    map(zip(g.row, g.col)) do (r, c)
        r / (1 + exp(-h.α * c))
    end
end

function decide_probs(h::CellHeuristic, game::Game)
    cv = cell_values(h, g)
    sum(softmax(h.λ * cv), dims=2)
end

g = Game(3, 1.)
h = CellHeuristic(1., 3.)
decide_probs(h, g)
#%%
