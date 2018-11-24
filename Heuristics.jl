import Distributions: MvNormal
import Distributions
import StatsBase: sample, Weights
import Statistics: mean
import Base
import Base: ==, hash
import DataStructures: OrderedDict
using Optim
using BenchmarkTools
import Printf: @printf, @sprintf
# import Base: rand

function softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

function weighted_softmax(priors, vals)
    ex = priors .* exp.(vals)
    ex ./= sum(ex)
    ex
end

sigmoid(x) = (1. / (1. + exp(-x)))

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

# These two are needed for dictionary to work with transposed games...
==(g::Game,k::Game) = (g.row == k.row) && (g.col == k.col)
Base.hash(g::Game) = hash(vcat(g.row, g.col))

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


# %% ==================== Abstract Heuristic ====================

abstract type Heuristic end


function play_distribution(h::Heuristic, g::Game)
    error("Unimplemented")
end

function Base.size(h::Heuristic)
    length(get_parameters(h))
end

function rand_params(h::Heuristic)
    error("Unimplemented")
end

function set_parameters!(h::Heuristic, x_vec::Vector{T} where T <: Real)
    for (field, x) in zip(fieldnames(typeof(h)), x_vec)
        setfield!(h, field, x)
    end
end

function get_parameters(h::Heuristic)
    fields = fieldnames(typeof(h))
    x_vec = [getfield(h, field) for field in fields]
end

function expected_payoff(h::Heuristic, opponent::Heuristic, g::Game)
    p = play_distribution(h, g)
    p_opp = play_distribution(opponent, transpose(g))
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end

mutable struct Costs
    α::Float64
    λ::Float64
    row::Float64
    level::Float64
    m_λ::Float64
end

## By using a mutable struct and this constructor we can initialize
## the cost with a flexible number of parameters
function Costs(;kwargs...)
    costs = Costs(zeros(length(fieldnames(Costs)))...)
    for v in kwargs
        setfield!(costs, v[1], v[2])
    end
    return costs
end
Costs(; α=1., λ=2.4)

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
     (h.α) ^2 * 0.0001 +  # TODO: these are basically regularizers: do they need to be so high?
     (h.γ) ^2 * 0.0001 +
     c.row)   # Answer: Not really, think I was just trying to tinker away an error in the optim.
end

# @assert Costs(;α=1., λ=1.)(RowHeuristic(1, 1, 1)) == 1.66211715726001


# %% ==================== SimHeuristic ====================

mutable struct SimHeuristic <: Heuristic
    h_list::Vector{Heuristic}
    level::Int64
end
SimHeuristic(hs::Vector) = SimHeuristic(hs, length(hs))

function get_parameters(s::SimHeuristic)
    x_vec = [x for h in s.h_list for x in get_parameters(h)]
end

function set_parameters!(s::SimHeuristic, x::Vector{T} where T <: Real)
    params = [(h, field) for h in s.h_list for field in fieldnames(typeof(h))]
    for ((h, field), x) in zip(params, x)
        setfield!(h, field, x)
    end
end

function play_distribution(s::SimHeuristic, g::Game)
    self_g = deepcopy(g)
    opp_g = deepcopy(transpose(g))
    probs = zeros(size(self_g))
    for i in 1:s.level
        if i == s.level
            probs = play_distribution(s.h_list[i], self_g)
        elseif (s.level - i) % 2 == 1
            opp_pred = play_distribution(s.h_list[i], opp_g)
            self_g.row .*= (size(game) .* opp_pred')
        else
            self_pred = play_distribution(s.h_list[i], self_g)
            opp_g.row .*= (size(game) .* self_pred')
        end
    end
    return probs
end

function cost(s::SimHeuristic, c::Costs)
    sum(map(c, s.h_list)) + c.level
end

# FIXME this doesn't work for some reason...
# Works for me!
game = Game(2, 0.)
sh = SimHeuristic([RowHeuristic(0, 0, 10), RowHeuristic(0, 0, 10)])
cost(sh, Costs(;α=0.01, λ=0.1))

# %% ==================== CacheHeuristic ====================

mutable struct CacheHeuristic <: Heuristic
    cache::Dict{Game, Vector{Float64}}
end

function CacheHeuristic(games::Vector{Game}, plays::Vector{Vector{Float64}})
    cache = Dict()
    for (game, play) in zip(games, plays)
        cache[game] = play
    end
    CacheHeuristic(cache)
end

function play_distribution(h::CacheHeuristic, g::Game)
    h.cache[g]
end

function cost(h::CacheHeuristic, cost::Costs)
    0.
end


# %% ==================== CellHeuristic ====================

mutable struct CellHeuristic <: Heuristic
    α::Real  # we might get a performance boost by using a parametric typem
    λ::Real
end

function play_distribution(h::CellHeuristic, g::Game)
    cell_values = zeros(Real, size(g), size(g))
    for i in 1:size(g), j in 1:size(g)
        r = @view g.row[i,j]
        c = @view g.col[i,j]
        cell_values[i,j] = r / (1 + exp(-h.α * c))
    end
    cell_probs = softmax(cell_values .* h.λ)
    [+(cell_probs[i,:]...) for i in 1:size(g)]
end

function cost(h::CellHeuristic, c::Costs)
    (abs(h.λ) * c.λ +
     2 * (sigmoid((h.α)^2) - 0.5) * c.α +
     (h.α) ^2 * 0.0001)
end

# %% ==================== MetaHeuristic ====================


mutable struct MetaHeuristic <: Heuristic
    h_list::Vector{Heuristic}
    prior::Vector{T} where T <: Real
end

function h_distribution(mh::MetaHeuristic, g::Game, opp_h::Heuristic, costs::Costs)
    h_values = map(h -> expected_payoff(h, opp_h, g) - costs(h), mh.h_list)
    softmax((mh.prior .+ h_values) ./ costs.m_λ)
    # weighted_softmax(mh.prior, h_values ./ mh.m_λ)
end

function play_distribution(mh::MetaHeuristic, g::Game, opp_h::Heuristic, costs::Costs)
    h_dist = h_distribution(mh, g, opp_h, costs)
    play = zeros(Real, size(g))
    for i in 1:length(h_dist)
        play += h_dist[i] * play_distribution(mh.h_list[i], g)
    end
    play
end


function get_parameters(mh::MetaHeuristic)
    res = deepcopy(mh.prior)
    x_vec = [x for h in mh.h_list for x in get_parameters(h)]
    push!(res, x_vec...)
    res
end


function set_parameters!(mh::MetaHeuristic, x::Vector{T} where T <: Real)
    setfield!(mh, :prior, x[1:length(mh.prior)])
    idx = length(mh.prior) + 1
    for h in mh.h_list
        set_parameters!(h, x[idx:(idx+size(h) -1)])
        idx = idx+size(h)
    end
end

function expected_payoff(h::MetaHeuristic, opponent::Heuristic, g::Game, costs::Costs)
    p = play_distribution(h, g, opp_h, costs)
    p_opp = play_distribution(opponent, transpose(g))
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end

# TODO: prior distribution on Heuristic weights
# - cost of deviating from prior for a specific game, kl_c * Kullback-Leibler divergence
#   this gives decision weigths as in the Rational Attention model of (Matějka, McKay 2015)
#   either we use priors as in (13), or we calculate the average payoffs as in (1).
# - optimize prior for an environment (list of games)


# %% ==================== Optimize_Heuristic ====================

function perf(h::Heuristic, games::Vector{Game}, opp_h::Heuristic, costs::Costs)
    payoff = 0
    for game in games
        payoff += expected_payoff(h, opp_h, game)
    end
    return (payoff/length(games) - costs(h))
end

function perf(mh::MetaHeuristic, games::Vector{Game}, opp_h::Heuristic, costs::Costs)
    payoff = 0
    for game in games
        cost = sum(h_distribution(mh, game, opp_h, costs) .* costs.(mh.h_list))
        payoff += expected_payoff(mh, opp_h, game, costs) - cost
    end
    return payoff/length(games)
end


function mean_square(x, y)
    sum((x - y).^2)
end

function likelihood(x,y)
    l = 0
    for (x_el, y_el) in zip(x,y)
        l += y_el*log(x_el)
    end
    -l
end

function prediction_loss(h::Heuristic, games::Vector{Game}, actual::Heuristic; loss_f = mean_square)
    loss = 0
    for game in games
        pred_p = play_distribution(h, game)
        actual_p = play_distribution(actual, game)
        loss += loss_f(pred_p, actual_p)
    end
    loss/length(games)
end

function prediction_loss(h::MetaHeuristic, games::Vector{Game}, actual::Heuristic, opp_h::Heuristic, costs::Costs; loss_f = mean_square)
    loss = 0
    for game in games
        pred_p = play_distribution(h, game, opp_h, costs)
        actual_p = play_distribution(actual, game)
        loss += loss_f(pred_p, actual_p)
    end
    loss/length(games)
end

function fit_h!(h::Heuristic, games::Vector{Game}, actual::Heuristic; init_x=nothing, loss_f=mean_square)
    if init_x == nothing
        init_x = ones(size(h))*0.01
    end
    function loss_wrap(x)
        set_parameters!(h, x)
        prediction_loss(h, games, actual; loss_f=loss_f)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    set_parameters!(h, x)
    h
end


function optimize_h!(h::Heuristic, games::Vector{Game}, opp_h::Heuristic, costs; init_x=nothing)
    if init_x == nothing
        init_x = ones(size(h))*0.01
    end
    function loss_wrap(x)
        set_parameters!(h, x)
        -perf(h, games, opp_h, costs)
    end
    # x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    set_parameters!(h, x)
    h
end


# %% Files for reading from files
function games_from_json(file_name)
        games_json = ""
        open(file_name) do f
                games_json = read(f, String)
        end
        games_vec = Game[]
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

function rand_costs(mins=[0.01, 0.01, 0.01, 0.01, 0.1], max=[0.2, 0.2, 0.5, 0.5, 1.])
    Costs([rand()*(max[i] - mins[i]) + mins[i] for i in 1:5]...)
end

# function opt_cost(h::Heuristic, games::Vector{Game}, opp_h::Heuristic, costs; loss_f = mean_square)

#
# mh = MetaHeuristic([RowHeuristic(0.3, -0.2, 2.), CellHeuristic(0.6, 3.), SimHeuristic([RowHeuristic(0.3, -0.2, 2.), RowHeuristic(0.3, -0.2, 2.)])], [0.3, 0.3, 0.4])
#
#
# mh2 = MetaHeuristic([RowHeuristic(0.3, -0.2, 2.), CellHeuristic(0.6, 3.)], [0.5, 0.5], 0.5)
#
# sh = SimHeuristic([RowHeuristic(0., 0., 0.), RowHeuristic(0., 0., 0.)])
# rh = RowHeuristic(0.,0.,0.)
# ch = CellHeuristic(0.,0.)
# opp_h = RowHeuristic(0., 0., 2.)
# opt_h = SimHeuristic([opp_h, RowHeuristic(0., 0., 10.)])
# games = [normalize(Game(3, -0.5)) for i in 1:100]
# push!(games, [normalize(Game(3, 0.8)) for i in 1:100]...)
# costs = Costs(;α=0.05, λ=0.1, row=0.05, level=0.03, m_λ=0.5)
# mh = MetaHeuristic([RowHeuristic(0.3, -0.2, 2.), CellHeuristic(0.6, 3.), SimHeuristic([RowHeuristic(0.3, -0.2, 2.), RowHeuristic(0.3, -0.2, 2.)])], [0.3, 0.3, 0.4], 0.3)
#
# optimize_h!(mh, games, opp_h, costs)
# optimize_h!(mh2, games, opp_h, costs)
# optimize_h!(sh, games, opp_h, costs)
# optimize_h!(rh, games, opp_h, costs)
# optimize_h!(ch, games, opp_h, costs)
# perf(mh, games, opp_h, costs)
# perf(mh.h_list[1], games, opp_h, costs)
# perf(rh, games, opp_h, costs)
# perf(ch, games, opp_h, costs)
# perf(opt_h, games, opp_h, costs)
#
# prediction_loss(mh, games, opt_h, opp_h, costs)
#
#
#
#
#
#
# games = [normalize(Game(3, -0.5)) for i in 1:100]
# plays = [play_distribution(opp_h, game) for game in games]
# cache_h = CacheHeuristic(games, plays)
#
#
# prediction_loss(ch, games, cache_h; loss_f = likelihood)
# prediction_loss(ch, games, cache_h)
# h = RowHeuristic(0., 0., 0.)
# ch = fit_h!(ch, games, cache_h; loss_f = mean_square)
#
#
# opp_h
#
# get_parameters(mh)
# set_parameters!(mh, [0.3, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 2., ])
#
#
# opp_h = RowHeuristic(0., 0., 2.)
# games = [normalize(Game(3, -0.5)) for i in 1:100]
# costs = Costs(;α=0.05, λ=0.01)
#
# play_distribution(mh::MetaHeuristic, games[1], opp_h, costs)
#
# game = games[1]
# h_values = map(h -> expected_payoff(h, opp_h, game) - costs(h), mh.h_list)
# h_distribution(mh, game, opp_h, costs)
# play_distribution(mh, game, opp_h, costs)
# play_distribution(mh.h_list[2], game)
# expected_payoff(mh, opp_h, game, costs) - costs(mh.h_list[2])
# expected_payoff(mh.h_list[2], opp_h, game)
#
# g = game
# h_dist = h_distribution(mh, g, opp_h, costs)
# play = zeros(Real, size(g))
# play_distribution(mh.h_list[i], g)
# i = 3
# play += h_dist[i] * play_distribution(mh.h_list[i], g)
#
#
# perf(mh, games, opp_h, costs)
#
# mh.prior = [1/3, 1/3, 1/3]
# mh.m_λ = 0.01
#
# sum(h_distribution(mh, game, opp_h, costs) .* costs.(mh.h_list))
#
# h_values = map(h -> expected_payoff(h, opp_h, game) - costs(h), mh.h_list)
# h_values = map(h -> expected_payoff(h, opp_h, game), mh.h_list)
#
# h_distribution(mh, game, opp_h, costs)
# costs(mh.h_list[1])
