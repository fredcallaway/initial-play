import Distributions: MvNormal
import Distributions
import StatsBase: sample, Weights
import Statistics: mean
import Base
import DataStructures: OrderedDict
using BlackBoxOptim
import Optim
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

function relative_values(h::Heuristic, game::Game)
    map(1:size(game)) do i
        μ_r = μ(game.row)
        μ_c = μ(game.col)
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
        elseif (s.level - 1) % 2 == 1
            opp_pred = decide_probs(s.h_list[i], opp_g)
            self_g.row .*= opp_pred'
        elseif (s.level - 1) % 2 == 0
            self_pred = decide_probs(s.h_list[i], self_g)
            opp_g.row .*= self_pred'
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


opp_cols_played = (opp_h, games) -> [g.row[:,decide(opp_h, transpose(g))] for g in games]

function opt_h!(h::Heuristic, opp_h::Union{Heuristic,Vector{Vector{Real}}}, games, h_dists; max_time=20.0, method=:de_rand_1_bin, trace=:silent)
    if isa(opp_h, Heuristic)
        opp_plays = opp_cols_played(opp_h, games)
    else
        opp_plays = opp_h
    end
    param_names = [:α, :γ, :λ]
    init_params = [h.α, h.γ, h.λ]
    function opt_fun(params)
        for (name, param) in zip(param_names, params)
            setfield!(h, name, param)
        end
        return -payoff(h, opp_plays, games)
    end
    res = bboptimize(opt_fun; SearchRange = [(h_dists["α"].a ,h_dists["α"].b), (h_dists["γ"].a ,h_dists["γ"].b), (h_dists["λ"].a ,h_dists["λ"].b)], MaxTime=max_time, Method=method, TraceMode = trace, TraceInterval =4.0)
    for (name, param) in zip(param_names, best_candidate(res))
        setfield!(h, name, param)
    end
end

function opt_s!(s::SimHeuristic, opp_h::Union{Heuristic,Vector{Vector{Real}}}, games, h_dists; max_time=20.0, method=method)
    if isa(opp_h, Heuristic)
        opp_plays = opp_cols_played(opp_h, games)
    else
        opp_plays = opp_h
    end
    n_h_params = length(fieldnames(Heuristic))
    param_names = repeat([:α, :γ, :λ], outer=[s.level])
    h = s.h_list[1]
    init_params = repeat([h.α, h.γ, h.λ], outer=[s.level])
    function opt_fun(params)
        for i in 1:s.level
            for j in 1:n_h_params
                param_idx = (i-1)*n_h_params + j
                setfield!(s.h_list[i], param_names[j], params[param_idx])
            end
        end
        return -payoff(s, opp_plays, games)
    end
    res = bboptimize(opt_fun; SearchRange = repeat([(h_dists["α"].a ,h_dists["α"].b), (h_dists["γ"].a ,h_dists["γ"].b), (h_dists["λ"].a ,h_dists["λ"].b)], outer=[s.level]), MaxTime=max_time,  Method=method, TraceMode = :silent, TraceInterval =4.0)
    res_params = best_candidate(res)
    for i in 1:s.level
        for j in 1:n_h_params
            param_idx = (i-1)*n_h_params + j
            setfield!(s.h_list[i], param_names[j], res_params[param_idx])
        end
    end
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
