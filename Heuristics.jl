import Distributions: MvNormal
import Distributions
import StatsBase: sample, Weights
import Statistics: mean
import Base
import DataStructures: OrderedDict
using Optim
using JSON
import Printf: @printf, @sprintf
# import Base: rand

function softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

sigmoid(x) = (1. ./ (1. .+ exp.(-x)))

# %% ==================== Games ====================

function sample_cell(ρ; max=9.5, min=-0.5, µ=5, σ=5)
    if ρ < 1 && ρ > -1
        Σ = [[1 ρ]
             [ρ 1]]
        dist = MvNormal(Σ)
        r, c = rand(dist)*σ .+ μ
        while any([r, c] .< min) || any([r, c] .> max)
             r, c = rand(dist)*σ .+ μ
        end
    else
        r = randn()*σ + μ
        c = ρ*(r - μ) + μ
        while any([r, c] .< min) || any([r, c] .> max)
            r = randn()*σ + μ
            c = ρ*(r - μ) + μ
        end
    end
    r = round(r)
    c = round(c)
    return (r, c)
end

struct Game
    row::Matrix{Real}
    col::Matrix{Real}
end

Game(size::Int, ρ::Number) = begin
    X = [sample_cell(ρ) for i in 1:size, j in 1:size]
    Game(
        map(x->x[1], X),
        map(x->x[2], X),
    )
end

Base.transpose(g::Game) = Game(transpose(g.col), transpose(g.row))
Base.size(g::Game) = size(g.row)[1]

function normalize(g::Game)
    Game(g.row .- mean(g.row), g.col .- mean(g.col))
end

Base.show(io::IO, g::Game) = begin
    for i in 1:size(g)
        for j in 1:size(g)
            # print(Int(g.row[i,j]), ",", Int(g.col[i,j]), "  ")
            @printf("%+3.1f , %+3.1f | ", g.row[i,j], g.col[i,j])
        end
        println()
    end
end

Game(3, 0.)


# %% ==================== Abstract Heuristic ====================
abstract type Heuristic end


function play_distribution(h::Heuristic, g::Game)
    error("Unimplemented")
end

function Base.size(h::Heuristic)
    # Returns the number of real valued parameters
    error("Unimplemented")
end

function rand_params(h::Heuristic)
    error("Unimplemented")
end

function set_parameters(h::Heuristic, x_vec::Vector{Real})
    error("Unimplemented")
end

function expected_payoff(h::Heuristic, opponent::Heuristic, g::Game)
    p = play_distribution(h, g)
    p_opp = play_distribution(opponent, g)
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end

struct Costs
    α::Float64
    λ::Float64
end

function cost(h::Heuristic, c::Costs)
    error("Unimplemented")
end
(c::Costs)(h::Heuristic) = cost(h, c)


# %% ==================== RowHeuristic ====================

mutable struct RowHeuristic <: Heuristic
    α::Real  # we might get a performance boost by using a parametric typem
    γ::Real
    λ::Real
end

function row_values(h::RowHeuristic, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        c = @view g.col[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        s = @. r / (1 + exp(-h.α * c))
        k = h.γ / max(1., (maximum(s) - minimum(s)))
        v = s' * softmax(k * s)
    end
end

function play_distribution(h::RowHeuristic, g::Game)
    softmax(h.λ * row_values(h, g))
end

function cost(h::RowHeuristic, c::Costs)
    (abs(h.λ) * c.λ +
     2 * (sigmoid((h.α)^2) - 0.5) * c.α +
     (h.α) ^2 * 0.1 +  # TODO: these are basically regularizers: do they need to be so high?
     (h.γ) ^2 * 0.1)
end

# @assert Costs(1., 1.)(RowHeuristic(1, 1, 1)) == 1.66211715726001

# %% ==================== SimHeuristic ====================
mutable struct SimHeuristic <: Heuristic
    h_list::Vector{Heuristic}
    level::Int64
end

# %% ==================== CacheHeuristic ====================
struct CacheHeuristic <: Heuristic
    cache::Dict{Game, Vector{Float64}}
end


struct MetaHeuristic <: Heuristic
end
# TODO: prior distribution on Heuristic weights
# - cost of deviating from prior for a specific game
# - optimize prior for an environment (list of games)
