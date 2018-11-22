import Distributions: MvNormal
import Distributions
import StatsBase: sample, Weights
import Statistics: mean
import Base
import DataStructures: OrderedDict
using Optim
using JSON
using BenchmarkTools
import Printf: @printf, @sprintf
# import Base: rand

function softmax(x)
    ex = exp.(x)
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
    γ::Float64
    λ::Float64
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
     (h.γ) ^2 * 0.0001)   # Answer: Not really, think I was just trying to tinker away an error in the optim.
end

# @assert Costs(;α=1., λ=1.)(RowHeuristic(1, 1, 1)) == 1.66211715726001


# %% ==================== SimHeuristic ====================

mutable struct SimHeuristic <: Heuristic
    h_list::Vector{Heuristic}
    level::Int64
end
SimHeuristic(hs::Vector) = SimHeuristic(hs, length(hs))

s = SimHeuristic([RowHeuristic(1., 2., 3.), RowHeuristic(4., 5., 6.)])


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
    sum(map(c, s.h_list))
end

# FIXME this doesn't work for some reason...
# Works for me!
game = Game(2, 0.)
sh = SimHeuristic([RowHeuristic(0, 0, 10), RowHeuristic(0, 0, 10)])
cost(sh, Costs(;α=0.01, γ=-0.01, λ=0.1))

# %% ==================== CacheHeuristic ====================

struct CacheHeuristic <: Heuristic
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


struct MetaHeuristic <: Heuristic
end
# TODO: prior distribution on Heuristic weights
# - cost of deviating from prior for a specific game
# - optimize prior for an environment (list of games)

# %% ==================== Optimize_Heuristic ====================

function perf(h::Heuristic, games::Vector{Game}, opp_h::Heuristic, costs::Costs)
    payoff = 0
    for game in games
        payoff += expected_payoff(h, opp_h, game)
    end
    return (payoff/length(games) - costs(h))
end

pay = 0
for i in eachindex(games)
    p = decide_probs(h, h2, α, games[i])
    pay +=  sum( (p - self_probs[i]).^2)
end
pay += sum(pred_cost(h) for h in h.h_list)
(pay/length(games))

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

function fit_h(h::Heuristic, games::Vector{Game}, actual::Heuristic; init_x=nothing, loss_f=mean_square)
    if init_x == nothing
        init_x = ones(size(h))*0.4
    end
    function loss_wrap(x)
        set_parameters!(h, x)
        prediction_loss(h, games, actual; loss_f=loss_f)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    set_parameters!(h, x)
    h
end

function optimize_h(h::Heuristic, games::Vector{Game}, opp_h::Heuristic, costs; init_x=nothing)
    if init_x == nothing
        init_x = ones(size(h))*0.4
    end
    function loss_wrap(x)
        set_parameters!(h, x)
        -perf(h, games, opp_h, costs)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    set_parameters!(h, x)
    h
end


h = SimHeuristic([RowHeuristic(0., 0., 0.), RowHeuristic(0., 0., 0.)])
opp_h = RowHeuristic(0., 0., 5.)
opt_h = SimHeuristic([opp_h, RowHeuristic(0., 0., 10.)])
games = [normalize(Game(3, -0.5)) for i in 1:1000]
costs = Costs(;α=0.05, λ=0.01)

optimize_h(h, games, opp_h, costs)
perf(h, games, opp_h, costs)
perf(opt_h, games, opp_h, costs)



games = [normalize(Game(3, -0.5)) for i in 1:100]
plays = [play_distribution(opp_h, game) for game in games]
ch = CacheHeuristic(games, plays)


prediction_loss(h, games, ch; loss_f = likelihood)
prediction_loss(h, games, ch)
h = RowHeuristic(0., 0., 0.)
h = fit_h(h, games, ch; loss_f = likeihood)
h

opp_h
