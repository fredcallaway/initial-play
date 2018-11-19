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
            # print(Int(g.row[i,j]), ",", Int(g.col[i,j]), "  ")
            @printf("%.1f ,%.1f | ", g.row[i,j], g.col[i,j])
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
        μ_r = μ(game.row)
        μ_c = μ(game.col)
        r = game.row[i, :] .- μ_r
        c = game.col[i, :] .- μ_c
        a = h.α / max(1., (maximum(c) - minimum(c)))
        # a = h.α
        s = map((r, c) -> r / (1 + exp(-a * c)), r, c)
        k = h.γ / max(1., (maximum(s) - minimum(s)))
        v = s' * softmax(k * s)
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
    cost += 2(sigmoid((h.α)^2) - 0.5) * c.α
    cost += (h.α)^2 *0.1
    cost += (h.γ)^2 *0.1
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

function loss_from_dist(h::SimHeuristic, games, opp_probs, costs::Costs)
    pay = 0
    for i in eachindex(games)
        p = decide_probs(h, games[i])
        for j in 1:size(games[i])
            pay += (p' * games[i].row[:,j]) * opp_probs[i][j]
        end
    end
    -(pay/length(games) - sum(cost(h, costs) for h in h.h_list))
end
function loss_from_dist(x::Vector{T} where T <: Real, games, opp_probs, costs)
    loss_from_dist(SimHeuristic(x), games, opp_probs, costs)
end

function pred_cost(h::Heuristic)
    cost = (h.λ)^2*0.001
    cost += (h.α)^2 *0.001
    cost += (h.γ)^2 *0.001
    cost
end

function pred_loss(h::SimHeuristic, games, self_probs, costs::Costs)
    pay = 0
    for i in eachindex(games)
        p = decide_probs(h, games[i])
        pay +=  sum( (p - self_probs[i]).^2)
    end
    pay += sum(pred_cost(h) for h in h.h_list)
    (pay/length(games))
end
function pred_loss(x::Vector{T} where T <: Real, games, self_probs, costs)
    pred_loss(SimHeuristic(x), games, self_probs, costs)
end


function pred_loss(h::SimHeuristic, h2::SimHeuristic, α, games, self_probs)
    pay = 0
    for i in eachindex(games)
        p = decide_probs(h, h2, α, games[i])
        pay +=  sum( (p - self_probs[i]).^2)
    end
    pay += sum(pred_cost(h) for h in h.h_list)
    (pay/length(games))
end

function pred_likelihood(h::SimHeuristic, games, self_probs, costs::Costs)
    like = 0
    for i in eachindex(games)
        p = decide_probs(h, games[i])
        for j in eachindex(p)
            like +=  self_probs[i][j] * log(p[j])
        end
    end
    like += sum(pred_cost(h) for h in h.h_list)
    (-like/length(games))
end
function pred_likelihood(x::Vector{T} where T <: Real, games, self_probs, costs)
    pred_likelihood(SimHeuristic(x), games, self_probs, costs)
end

function optimize_h(level, games, opp_plays, costs; init_x=nothing, loss_f = loss)
    loss_wrap(x) = loss_f(x, games, opp_plays, costs)
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
(b::Bounds)(noise) = vcat((b.lower .+ reshape(noise, 3, :) .* (b.upper .- b.lower))...)
Base.rand(b::Bounds) = b(rand(3))
Base.rand(b::Bounds, level::Int64) = b(rand(3*level))

function sample_init(n, level)
    n -= 1 # because we add 0.1s later
    X = (LHCoptim(n, 3*level, 1000)[1] .- 1) ./ n
    init = [bounds(X[i, :]) .+ 0.001 for i in 1:size(X)[1]]
    push!(init, 0.001 * ones(3*level))
end


function games_from_json(file_name)
        games_json = ""
        open(file_name) do f
                games_json = read(f, String)
        end
        games_vec = []
        games_json = JSON.parse(games_json)
        for g in games_json
                row = [g["row"][j][i] for i in 1:length(g["row"][1]), j in 1:length(g["row"])]
                col = [g["col"][j][i] for i in 1:length(g["col"][1]), j in 1:length(g["col"])]
                game = Game(row, col)
                push!(games_vec, game)
        end
        return games_vec
end

function plays_vec_from_json(file_name)
        json_string = ""
        open(file_name) do f
                json_string = read(f, String)
        end
        loaded_vec = JSON.parse(json_string)
        convert(Array{Array{Float64,1},1}, loaded_vec)
end


function sample_init(n, level)
    n -= 1 # because we add 0.1s later
    X = (LHCoptim(n, 3*level, 1000)[1] .- 1) ./ n
    init = [bounds(X[i, :]) .+ 0.001 for i in 1:size(X)[1]]
    push!(init, 0.001 * ones(3*level))
end

function decide_probs(s1::SimHeuristic, s2::SimHeuristic, α::Float64, game::Game)
    s1_pred = decide_probs(s1, game)
    s2_pred = decide_probs(s2, game)
    pred = α*s1_pred .+ (1-α)s2_pred
end

function pred_loss(h::SimHeuristic, h2::SimHeuristic, α, games, self_probs)
    pay = 0
    for i in eachindex(games)
        p = decide_probs(h, h2, α, games[i])
        pay +=  sum( (p - self_probs[i]).^2)
    end
    pay += sum(pred_cost(h, costs) for h in h.h_list)
    (pay/length(games))
end

function costs_preds(costs_1, costs_2, games, row_plays, col_plays)
    s1 = optimize_h(1, games, col_plays, costs_1; init_x=[0.,0.,0.], loss_f = loss_from_dist)
    s2 = optimize_h(2, games, col_plays, costs_2; loss_f = loss_from_dist)
    function α_fun(α)
         pred_loss(s1, s2, α, games, row_plays)
    end
    α = Optim.minimizer(optimize(α_fun, 0., 1.))
    perf = α_fun(α)
    return (perf, α, s1, s2)
end

function costs_preds(x::Vector, games, row_plays, col_plays)
    costs_1 = Costs(x[1], x[2])
    costs_2 = Costs(x[3], x[4])
    costs_preds(costs_1, costs_2, games, row_plays, col_plays)
end


function costs_preds(costs_1, costs_2, α::Float64, games, row_plays, col_plays)
    s1 = optimize_h(1, games, col_plays, costs_1; init_x=[0.,0.,0.], loss_f = loss_from_dist)
    s2 = optimize_h(2, games, col_plays, costs_2; loss_f = loss_from_dist)
    perf = pred_loss(s1, s2, α, games, row_plays)
    return (perf, α, s1, s2)
end
function costs_preds(x::Vector, α::Float64, games, row_plays, col_plays)
    costs_1 = Costs(x[1], x[2])
    costs_2 = Costs(x[3], x[4])
    costs_preds(costs_1, costs_2, α, games, row_plays, col_plays)
end

function rand_costs(α_l, α_u, λ_l, λ_u)
    x = zeros(4)
    x[1] = rand()*(α_u - α_l) + α_l
    x[2] = rand()*(λ_u - λ_l) + λ_l
    x[3] = rand()*(α_u - α_l) + α_l
    x[4] = rand()*(λ_u - λ_l) + λ_l
    return x
end
