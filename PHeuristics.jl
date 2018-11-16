import Distributions: MvNormal
import Distributions
import StatsBase: sample, Weights
import Statistics: mean
import Base
import DataStructures: OrderedDict
using BlackBoxOptim
using Optim
import Printf: @printf, @sprintf

function softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

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

Base.show(io::IO, g::Game) = begin
    for i in 1:size(g)
        for j in 1:size(g)
            print(Int(g.row[i,j]), ",", Int(g.col[i,j]), "  ")
        end
        println()
    end
end

mutable struct Heuristic
    α::Real
    γ::Real
    λ::Real
end

mutable struct SimHeuristic
    h_list::Vector{Heuristic}
    level::Int64
end


SimHeuristic(hs::Vector{Heuristic}) = SimHeuristic(hs, length(hs))

Heuristic(dists::OrderedDict) = Heuristic(map(rand, dists.vals)...)

SimHeuristic(dists::OrderedDict, level::Int64) = SimHeuristic([Heuristic(map(rand, dists.vals)...) for i in 1:level], level)

Base.show(io::IO, h::Heuristic) = @printf(io, "Heuristic: α=%.2f, γ=%.2f, λ=%.2f", h.α, h.γ, h.λ)

function μ(mat::Matrix{Real}, row::Int64=0)
    if row == 0
        return mean(mat)
    else
        return (mean(mat[row,:]))
    end
end

opp_cols_played = (opp_h, games) -> [g.row[:,decide(opp_h, transpose(g))] for g in games]


function relative_values(h::Heuristic, game::Game)
    map(1:size(game)) do i
        μ_r = μ(game.row, i)
        μ_c = μ(game.col, i)
        r = game.row[i, :] .- μ_r
        c = game.col[i, :] .- μ_c
        s = map((r, c) -> r / (1 + exp(-h.α * c)), r, c)
        v = s' * softmax(h.γ * s)
    end
end

function decide(h::Heuristic, game::Game)
    v = relative_values(h, game)
    v = softmax(h.λ * v)
    choice = sample(1:length(v), Weights(v))
end

function decide_probs(h::Heuristic, game::Game)
    v = relative_values(h, game)
    v = softmax(h.λ*v)
end


function decide(s::SimHeuristic, game::Game)
    self_g = deepcopy(game)
    opp_g = deepcopy(transpose(game))
    choice = 0
    for i in 1:s.level
        if i == s.level
            choice = decide(s.h_list[i], self_g)
        elseif (s.level - 1) % 2 == 1
            opp_pred = decide_probs(s.h_list[i], opp_g)
            self_g.row .*= opp_pred'
        elseif (s.level - 1) % 2 == 0
            self_pred = decide_probs(s.h_list[i], self_g)
            opp_g.row .*= self_pred'
        end
    end
    # println("Done, i=%f", i)
    return choice
end


function decide_probs(s::SimHeuristic, game::Game)
    self_g = deepcopy(game)
    opp_g = deepcopy(transpose(game))
    probs = zeros(size(self_g))
    for i in 1:s.level
        if i == s.level
            probs = decide_probs(s.h_list[i], self_g)
        elseif (s.level - i) % 2 == 1
            opp_pred = decide_probs(s.h_list[i], opp_g)
            self_g.row .*= (size(game) .* opp_pred')
        elseif (s.level - i) % 2 == 0
            self_pred = decide_probs(s.h_list[i], self_g)
            opp_g.row .*= (size(game) .* self_pred')
        end
    end
    # println("Done, i=%f", i)
    return probs
end


function payoff(h, opp_h::Union{Heuristic, Vector{Vector{Real}}}, games)
    payoff = 0
    if isa(opp_h, Heuristic)
        for g in games
            decision_p = decide_probs(h, g)
            opp_decision = decide(opp_h, transpose(g))
            payoff += decision_p' * g.row[decision, opp_decision]
        end
    else
        for i in 1:length(games)
            decision_p = decide_probs(h, games[i])
            payoff += decision_p' * opp_h[i]
        end
    end
    return payoff
end


function rand_heuristic_perf(h_dist::OrderedDict, opp_plays::Vector{Vector{Real}}, games::Vector{Game}, level::Int64=1)
    if level == 1
        h = Heuristic(h_dist)
    else
        h = SimHeuristic(h_dist, level)
    end
    fitness = payoff(h, opp_plays, games)
    return (fitness, h)
end


# ---------- Optimization

struct Costs
    α::Float64
    λ::Float64
end

function cost(h::Heuristic, c::Costs)
    cost = abs(h.λ) * c.λ
    # cost += 2(sigmoid(abs(5h.α)) - 0.5) * c.α
    cost += 2(sigmoid(5*sqrt((h.α)^2)) - 0.5) * c.α
    cost += sqrt((h.α)^2)*0.01
    cost += sqrt((h.γ)^2)*0.01
    cost
end
sigmoid(x) = (1. ./ (1. .+ exp.(-x)))


SimHeuristic(x::Vector{T} where T <: Real) = begin
     map(1:3:length(x)) do i
        Heuristic(x[i:i+2]...)
    end |> SimHeuristic
end

function loss(h::SimHeuristic, games, opp_plays, costs::Costs)
    pay = 0
    for i in eachindex(games)
        p = decide_probs(h, games[i])
        pay += p' * opp_plays[i]
    end
    -(pay / length(games) - sum(cost(h, costs) for h in h.h_list))
end
function loss(x::Vector{T} where T <: Real, games, opp_plays, costs)
    loss(SimHeuristic(x), games, opp_plays, costs)
end

function optimize_h(level, games, opp_plays, costs; init_x=nothing)
    loss_wrap(x) = loss(x, games, opp_plays, costs)
    if init_x == nothing
        init_x = ones(3 * level) * 0.1
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    SimHeuristic(x)
end
function optimize_h(level, ρ, game_size, opp_h, costs; n_games=1000)
    games = [Game(game_size, ρ) for i in range(1,n_games)]
    opp_plays = opp_cols_played(opp_h, games);
    optimize_h(level, games, opp_plays, costs)
end


struct Bounds
    lower::Vector{Float64}
    upper::Vector{Float64}
end
rand(b::Bounds) = b.lower .+ rand(3) .* (b.upper .- b.lower)
rand(b::Bounds, level::Int64) = vcat([rand(b) for i in 1:level]...)
