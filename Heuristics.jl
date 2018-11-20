
#%% Temporary
struct Game
    row::Matrix{Float64}
    col::Matrix{Float64}
end
Game() = Game(rand(2,2), rand(2,2))
#%%

abstract type Heuristic end

mutable struct RowHeuristic <: Heuristic
    α::Real  # we might get a performance boost by using a parametric type
    γ::Real
    λ::Real
end

mutable struct SimHeuristic <: Heuristic
    h_list::Vector{Heuristic}
    level::Int64
end

struct CacheHeuristic <: Heuristic
    cache::Dict{Game, Vector{Float64}}
end

function play_distribution(h::Heuristic, g::Game)
    error("Unimplemented")
end

function cost(h::Heuristic)
    error("Unimplemented")
end

function expected_payoff(h::Heurstic, opponent::Heuristic, g::Game)
    p = play_distribution(h, g)
    p_opp = play_distribution(opponent, g)
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end
