import Distributions: MvNormal
import Distributions
import StatsBase: sample, Weights
import Statistics: mean
import Base
import Base: ==, hash
import DataStructures: OrderedDict
using Optim
# using BenchmarkTools
import Printf: @printf, @sprintf
# import LineSearches
# using Flux # To not get a conflict with its sigmoid and my_softmax
# import Base: rand
function my_softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

function weighted_softmax(priors, vals)
    ex = priors .* exp.(vals)
    ex ./= sum(ex)
    ex
end

# sigmoid(x) = (1. / (1. + exp(-x)))

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
    # pritnln(g.name)
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

function expected_payoff(p, opponent::Heuristic, g::Game)
    p_opp = play_distribution(opponent, transpose(g))
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end

function expected_payoff(p::Vector, p_opp::Vector, g::Game)
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end

function expected_payoff(h::Heuristic, p_opp::Vector, g::Game)
    p = play_distribution(h, g)
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end


mutable struct Costs
    α::Float64
    λ::Float64
    # row::Float64
    level::Float64
    m_λ::Float64
    # pure::Float64
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

function cost(h::Heuristic, c::Costs)
    error("Unimplemented")
end

(c::Costs)(h::Heuristic) = cost(h, c)


# %% ==================== RowCellHeuristic ====================

mutable struct RowCellHeuristic <: Heuristic
    α::Real  # we might get a performance boost by using a parametric typem
    γ::Real
    λ::Real
end


function row_values(h::RowCellHeuristic, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        c = @view g.col[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        # s = @. r / (1 + exp(-h.α * c))
        s = @. r / (1 + exp(-h.α * c))
        k = h.γ / max(1., (maximum(s) - minimum(s)))
        v = s' * my_softmax(k * s)
    end
end

function play_distribution(h::RowCellHeuristic, g::Game)
    my_softmax(h.λ * row_values(h, g))
end
#
# function cost(h::RowCellHeuristic, c::Costs)
#     (abs(h.λ) * c.λ +
#      2 * (sigmoid((h.α)^2) - 0.5) * c.α +
#      # c.α +
#      (h.α) ^2 * 0.01 +  # TODO: these are basically regularizers: do they need to be so high?
#      (h.γ) ^2 * 0.01 +
#      c.row)   # Answer: Not really, think I was just trying to tinker away an error in the optim.
# end

# @assert Costs(;α=1., λ=1.)(RowHeuristic(1, 1, 1)) == 1.66211715726001

# %% ==================== RowHeuristic ====================

mutable struct RowHeuristic <: Heuristic
    γ::Real
    λ::Real
end


function row_values(h::RowHeuristic, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        k = h.γ / max(1., (maximum(r) - minimum(r)))
        v = r' * my_softmax(k * r)
    end
end

function play_distribution(h::RowHeuristic, g::Game)
    my_softmax(h.λ * row_values(h, g))
end

function cost(h::RowHeuristic, c::Costs)
    abs(h.λ) *c.λ + (h.γ) ^2 * 0.01
end
# function cost(h::RowHeuristic, c::Costs)
#     (abs(h.λ) * c.λ +
#      (h.γ) ^2 * 0.01 +
#      c.row)   # Answer: Not really, think I was just trying to tinker away an error in the optim.
# end

# %% ==================== RowγHeuristic ====================

mutable struct RowγHeuristic <: Heuristic
    γ::Real
    λ::Real
end


function row_values(h::RowγHeuristic, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        k = h.γ / max(1., (maximum(r) - minimum(r)))
        v = r' * my_softmax(k * r)
    end
end

function play_distribution(h::RowγHeuristic, g::Game)
    my_softmax(h.λ * row_values(h, g))
end

function cost(h::RowγHeuristic, c::Costs)
    abs(h.λ) *c.λ
end

function get_parameters(h::RowγHeuristic)
    [h.λ]
end

function set_parameters!(h::RowγHeuristic, x::Vector{T} where T <: Real)
    h.λ = x[1]
end


# function cost(h::RowHeuristic, c::Costs)
#     (abs(h.λ) * c.λ +
#      (h.γ) ^2 * 0.01 +
#      c.row)   # Answer: Not really, think I was just trying to tinker away an error in the optim.
# end


# %% ==================== RowMean ====================

mutable struct RowMean <: Heuristic
    λ::Real
end

function row_values(h::RowMean, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        v = mean(r)
    end
end

function play_distribution(h::RowMean, g::Game)
    my_softmax(h.λ * row_values(h, g))
end

function cost(h::RowMean, c::Costs)
    abs(h.λ) * c.λ
end


# %% ==================== RowMin ====================

mutable struct RowMin <: Heuristic
    λ::Real
end

function row_values(h::RowMin, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        v = minimum(r)
    end
end

function play_distribution(h::RowMin, g::Game)
    my_softmax(h.λ * row_values(h, g))
end

function cost(h::RowMin, c::Costs)
    (abs(h.λ) * c.λ)
end

# %% ==================== RowMax ====================

mutable struct RowMax <: Heuristic
    λ::Real
end

function row_values(h::RowMax, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        v = maximum(r)
    end
end

function play_distribution(h::RowMax, g::Game)
    my_softmax(h.λ * row_values(h, g))
end

function cost(h::RowMax, c::Costs)
    (abs(h.λ) * c.λ
     #+  c.row   # Answer: Not really, think I was just trying to tinker away an error in the optim.
     )
end

# %% ==================== RowJoint ====================

mutable struct RowJoint <: Heuristic
    λ::Real
end

function row_values(h::RowJoint, g::Game)
    map(1:size(g)) do i
        r = @view g.row[i, :]
        c = @view g.col[i, :]
        # a = h.α / max(1., (maximum(c) - minimum(c)))  # NOTE: Do we want this?
        v = mean(vcat(r, c))
    end
end

function play_distribution(h::RowJoint, g::Game)
    my_softmax(h.λ * row_values(h, g))
end

function cost(h::RowJoint, c::Costs)
    (abs(h.λ) * c.λ
     #+  c.row   # Answer: Not really, think I was just trying to tinker away an error in the optim.
     )
end



# %% ==================== RandomHeuristic ====================

mutable struct RandomHeuristic <: Heuristic
end

function play_distribution(h::RandomHeuristic, g::Game)
    ones(size(g))/size(g)
end


function cost(h::RandomHeuristic, c::Costs)
    0.
end

#%% ====================  Pure strategy heuristic ====================
mutable struct PureHeuristic <: Heuristic
    s::Int64
end

function get_parameters(s::PureHeuristic)
    []
end

function play_distribution(h::PureHeuristic, g::Game)
    res = zeros(size(g))
    res[h.s] = 1.
    res
end


# function cost(h::PureHeuristic, c::Costs)
#     c.pure
# end
function cost(h::PureHeuristic, c::Costs)
    0.
end

# %% ==================== MaxHeuristic ====================

mutable struct MaxHeuristic <: Heuristic
    λ::Real
end

function play_distribution(h::MaxHeuristic, g::Game)
    cell_values = zeros(Real, size(g), size(g))
    for i in 1:size(g), j in 1:size(g)
        # r = @view g.row[i,j]
        r = g.row[i,j]
        # c = @view g.col[i,j]
        # c = g.col[i,j]
        cell_values[i,j] = r
    end
    cell_probs = my_softmax(cell_values .* h.λ)
    [+(cell_probs[i,:]...) for i in 1:size(g)]
end


function cost(h::MaxHeuristic, c::Costs)
    abs(h.λ)*c.λ
end



# %% ==================== JointMax ====================

mutable struct JointMax <: Heuristic
    λ::Real
end

function play_distribution(h::JointMax, g::Game)
    cell_values = zeros(Real, size(g), size(g))
    for i in 1:size(g), j in 1:size(g)
        # r = @view g.row[i,j]
        r = g.row[i,j]
        # c = @view g.col[i,j]
        c = g.col[i,j]
        cell_values[i,j] = min(r,c)
    end
    cell_probs = my_softmax(cell_values .* h.λ)
    [+(cell_probs[i,:]...) for i in 1:size(g)]
end


function cost(h::JointMax, c::Costs)
    abs(h.λ)*c.α
end
# function cost(h::JointMax, c::Costs)
#     abs(h.λ)*c.λ + c.α
# end

# %% ==================== Nasheuristic ====================
function find_pure_nash(g::Game)
    neqs = []
    for j in 1:size(g)
        for i in filter(x -> g.row[x,j] == maximum(g.row[:,j]), 1:size(g))
            if g.col[i,j] == maximum(g.col[i,:])
                push!(neqs, (i,j, g.row[i,j]))
            end
        end
    end
    neqs
end

mutable struct NashHeuristic <: Heuristic
    λ::Real
end

function play_distribution(h::NashHeuristic, g::Game)
    neqs = find_pure_nash(g)
    pres = zeros(Real, size(g))
    if length(neqs) > 0
        res = my_softmax([h.λ * x[3] for x in neqs])
        for i in 1:length(res)
            pres[neqs[i][1]] += res[i]
        end
    end
    return(pres)
end

function cost(h::NashHeuristic, c::Costs)
    # c.nash +c.λ*abs(h.λ)
    # c.nash + 0.001*abs(h.λ)^2
    0.001*abs(h.λ)^2
end

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
sh = SimHeuristic([RowHeuristic(0, 10), RowHeuristic(0, 10)])
cost(sh, Costs(;m_λ=0.01, λ=0.1))

# %% ==================== CacheHeuristic ====================

mutable struct CacheHeuristic <: Heuristic
    cache::Dict{Game, Vector{T} where T <: Real}
end

function CacheHeuristic(games::Vector{Game}, plays::Vector{Vector{T}} where T <: Real)
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
    cell_probs = my_softmax(cell_values .* h.λ)
    [+(cell_probs[i,:]...) for i in 1:size(g)]
end

# function cost(h::CellHeuristic, c::Costs)
#     (abs(h.λ) * c.λ +
#      2 * (sigmoid((h.α)^2) - 0.5) * c.α +
#      # c.α +
#      (h.α) ^2 * 0.01)
# end

#%%  ==================== QLK Heuristic ====================
mutable struct QLK <: Heuristic
    α_0::Real
    α_1::Real
    λ1::Real
    λ21::Real
    λ22::Real
end

QLK(α_0, α_1, λ) = QLK(α_0, α_1, λ,λ, λ)
QLK(α_0, α_1, λ1, λ21, λ22) = QLK(α_0, α_1, λ1, λ21, λ22)

function play_distribution(h::QLK, g::Game)
    level_0 = ones(Real, size(g))/size(g)
    level_1 = play_distribution(RowHeuristic(0., h.λ1), g)
    level_2 = play_distribution(SimHeuristic([RowHeuristic(0., h.λ21), RowHeuristic(0., h.λ22)]), g)
    α_0 = min(max(h.α_0, 0.),1.)
    α_1 = min(max(h.α_1, 0.), 1.)
    α_2 = min(max(1 - α_0 - α_1, 0.), 1.)
    return level_0*α_0 + level_1*α_1 + level_2*α_2
end

function cost(h::QLK, c::Costs)
    α_0 = min(max(h.α_0, 0.),1.)
    α_1 = max(min(h.α_1, 1 - α_0), 0.)
    α_2 = min(max(1 - α_0 - α_1, 0.), 1.)
    (abs(h.λ1) * c.λ*α_1 +
    (abs(h.λ21) + abs(h.λ22)) *c.λ*(1 -α_1 - α_0) + c.level*(1 -α_1 - α_0))
end

#%%  ==================== QCH Heuristic ====================
mutable struct QCH <: Heuristic
    α_0::Real
    α_1::Real
    λ1::Real
    λ21::Real
    λ22::Real
end

QCH(α_0, α_1, λ) = QCH(α_0, α_1, λ, λ, λ)
QCH() = QCH(1., 1., 1.)

function play_distribution(h::QCH, g::Game)
    level_0 = ones(Real, size(g))/size(g)
    level_1 = play_distribution(RowHeuristic(0., h.λ1), g)
    opp_level_1 = play_distribution(RowHeuristic(0., h.λ21), transpose(g))
    opp_play = (level_0*h.α_0 + opp_level_1*h.α_1)/(h.α_0 + h.α_1)
    opp_h = CacheHeuristic([transpose(g)], [opp_play])
    level_2 = play_distribution(SimHeuristic([opp_h, RowHeuristic(0., h.λ22)]), g)
    # α_2 = max(1 - h.α_0 - h.α_1, 0.)
    α_0 = min(max(h.α_0, 0.),1.)

    α_1 = max(min(h.α_1, 1 - α_0), 0.)
    α_2 = min(max(1 - α_0 - α_1, 0.), 1.)
    return level_0*α_0 + level_1*α_1 + level_2*α_2
end

function cost(h::QCH, c::Costs)
    α_0 = min(max(h.α_0, 0.),1.)
    α_1 = max(min(h.α_1, 1 - α_0), 0.)
    α_2 = min(max(1 - α_0 - α_1, 0.), 1.)
    (abs(h.λ1) * c.λ*α_1 +
    (abs(h.λ21) + abs(h.λ22)) *c.λ*(1 -α_1 - α_0) + c.level*(1 -α_1 - α_0))
end
# %% ==================== MetaHeuristic ====================


mutable struct MetaHeuristic <: Heuristic
    h_list::Vector{Heuristic}
    prior::Vector{T} where T <: Real
end

function h_distribution(mh::MetaHeuristic, g::Game, opp_h::Heuristic, costs::Costs)
    h_values = map(h -> expected_payoff(h, opp_h, g) - costs(h), mh.h_list)
    my_softmax((mh.prior .+ h_values) ./ costs.m_λ)
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

function play_distribution(mh::MetaHeuristic, g::Game)
    ### Play according to prior,
    ### does not adjust according to performance in specific game
    h_dist = my_softmax(mh.prior ./ costs.m_λ)
    play = zeros(Real, size(g))
    for i in 1:length(h_dist)
        play += h_dist[i] * play_distribution(mh.h_list[i], g)
    end
    play
end


function get_parameters(mh::MetaHeuristic)
    # res = deepcopy(mh.prior)
    x_vec = [x for h in mh.h_list for x in get_parameters(h)]
    # push!(res, x_vec...)
    # res
end

function set_parameters!(mh::MetaHeuristic, x::Vector{T} where T <: Real)
    # setfield!(mh, :prior, x[1:length(mh.prior)])
    # idx = length(mh.prior) + 1
    idx = 1
    for h in mh.h_list
        set_parameters!(h, x[idx:(idx+size(h) -1)])
        idx = idx+size(h)
    end
end

function expected_payoff(h::MetaHeuristic, opponent::Heuristic, g::Game, costs::Costs)
    p = play_distribution(h, g, opponent, costs)
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

function perf(h::Heuristic, game::Game, opp_h::Vector, costs::Costs)
    return (expected_payoff(h, opp_h, game) - costs(h))
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
    l = sum(y .* log.(x))
    l = isinf(l) ?  sum(y .* (log.((x .+ 0.000001)./1.000003))) : l
    -l
end


function prediction_loss(h::Heuristic, games::Vector{Game}, actual::Heuristic; loss_f = likelihood)
    loss = 0
    for game in games
        pred_p = play_distribution(h, game)
        actual_p = play_distribution(actual, game)
        loss += loss_f(pred_p, actual_p)
    end
    loss/length(games)
end

function prediction_loss(h::MetaHeuristic, games::Vector{Game}, actual::Heuristic, opp_h::Heuristic, costs::Costs; loss_f = likelihood)
    loss = 0
    for game in games
        pred_p = play_distribution(h, game, opp_h, costs)
        actual_p = play_distribution(actual, game)
        loss += loss_f(pred_p, actual_p)
    end
    loss/length(games)
end

function fit_h!(h::Heuristic, games::Vector{Game}, actual::Heuristic; init_x=nothing, loss_f = likelihood)
    if init_x == nothing
        init_x = ones(size(h))*0.1
    end
    function loss_wrap(x)
        set_parameters!(h, x)
        prediction_loss(h, games, actual; loss_f=loss_f)
    end
    # x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS())) #TODO: get autodiff to work with logarithm in likelihood
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    set_parameters!(h, x)
    h
end

function fit_h!(h::MetaHeuristic, games::Vector{Game}, actual::Heuristic, opp_h, costs; init_x=nothing, loss_f = likelihood)
    if init_x == nothing
        init_x = ones(size(h))*0.01
    end
    function loss_wrap(x)
        set_parameters!(h, x)
        prediction_loss(h, games, actual, opp_h, costs; loss_f=loss_f)
    end
    # x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS())) # TODO: get autodiff to work with logarithm in likelihood
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
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
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    # x = Optim.minimizer(optimize(loss_wrap, init_x, LBFGS(;linesearch = LineSearches.MoreThuente()); autodiff = :forward))
    # x = Optim.minimizer(optimize(loss_wrap, init_x, Newton(); autodiff = :forward))
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
                row = [g["row"][j][i] + rand()*0.0001 for i in 1:length(g["row"][1]), j in 1:length(g["row"])]
                col = [g["col"][j][i] + rand()*0.0001 for i in 1:length(g["col"][1]), j in 1:length(g["col"])]
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
    Costs([rand()*(max[i] - mins[i]) + mins[i] for i in 1:length(mins)]...)
end

function fit_prior!(mh, games, actual_h, opp_h, costs)
    function wrap(x)
        mh.prior = x
        prediction_loss(mh, games, actual_h, opp_h, costs)
    end
    res = Optim.minimizer(optimize(wrap, copy(mh.prior), LBFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    mh.prior = res
    # println(prediction_loss(mh, games, actual_h, opp_h, costs))
    return mh
end

function opt_prior!(mh, games, opp_h, costs)
    function wrap(x)
        mh.prior = x
        -perf(mh, games, opp_h, costs)
    end
    res = Optim.minimizer(optimize(wrap, copy(mh.prior), LBFGS(), Optim.Options(time_limit=60); autodiff = :forward))
    mh.prior = res
    return mh
end


function mean(costs_vec::Vector{Costs})
    means = []
    for field in fieldnames(Costs)
        push!(means, mean([getfield(c, field) for c in costs_vec]))
    end
    means
end

function std(costs_vec::Vector{Costs})
    stds = []
    for field in fieldnames(Costs)
        push!(stds, std([getfield(c, field) for c in costs_vec]))
    end
    stds
end

function std_hdist(h, games, opp_h, best_costs)
    stds =  map(1:length(h.prior)) do i
        std([h_distribution(h, game, opp_h, best_costs)[i] for game in exp_games])
    end
end

function max_hdist(h, games, opp_h, best_costs)
    stds =  map(1:length(h.prior)) do i
        maximum([h_distribution(h, game, opp_h, best_costs)[i] for game in exp_games])
    end
end

function min_hdist(h, games, opp_h, best_costs)
    stds =  map(1:length(h.prior)) do i
        minimum([h_distribution(h, game, opp_h, best_costs)[i] for game in exp_games])
    end
end
