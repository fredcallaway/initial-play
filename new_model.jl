using LinearAlgebra: norm
using SplitApplyCombine
import Base.rand

include("Heuristics.jl")
include("DeepLayers.jl")
include("rule_learning.jl")

mutable struct Cost_Space
    α_min::Float64
    α_max::Float64
    λ_min::Float64
    λ_max::Float64
    level_min::Float64
    level_max::Float64
    m_λ_min::Float64
    m_λ_max::Float64
end

Cost_Space(α::Tuple{Float64, Float64}, λ::Tuple{Float64, Float64}, level::Tuple{Float64, Float64}, m_λ::Tuple{Float64, Float64}) = Cost_Space(α[1], α[2], λ[1], λ[2], level[1], level[2], m_λ[1], m_λ[2])
scale(x, low, high) = low + x * (high - low)
logscale(x, low, high) = exp(scale(x, log(low), log(high)))

function rand(cs::Cost_Space)
    α = logscale(rand(), cs.α_min, cs.α_max)
    λ = logscale(rand(), cs.λ_min, cs.λ_max)
    level = scale(rand(), cs.level_min, cs.level_max)
    m_λ = logscale(rand(), cs.m_λ_min, cs.m_λ_max)
    Costs(α,λ,level,m_λ)
end

function rand(cs::Cost_Space, n::Int64)
    [rand(cs) for i in 1:n]
end

function fit_model(base_model::QCH, data, costs::Costs=Costs(0.1,0.1,0.2,0.8))
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    try
        return fit_h!(deepcopy(base_model), games, empirical_play)
    catch
        deepcopy(base_model)
    end
end

function optimize_model(base_model::QCH, data, costs::Costs=Costs(0.1,0.1,0.2,0.8))
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    try
        return optimize_h!(deepcopy(base_model), games, empirical_play, costs)
    catch
        return deepcopy(base_model)
    end
end

function test(model, data)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play)
end


####################################################
# %% common Treatment: MetaHeuristic
####################################################
# mh_pos = MetaHeuristic([JointMax(3.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);


function fit_model(base_model::MetaHeuristic, data, costs::Costs; n_iter=5)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    for i in 1:n_iter
        fit_prior!(model, games, empirical_play, empirical_play, costs)
        fit_h!(model, games, empirical_play, empirical_play, costs)
    end
    model
end

function optimize_model(base_model::MetaHeuristic, data, costs::Costs; n_iter=5)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    for i in 1:n_iter
        optimize_h!(model, games, empirical_play, costs)
        opt_prior!(model, games, empirical_play, costs)
    end
    model
end

####################################################
# %%  DeepHeuristics
####################################################
# mh_pos = MetaHeuristic([JointMax(3.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);


mutable struct DeepCosts
    γ::Float64
    exact::Float64
    sim::Float64
end

(c::DeepCosts)(m::Chain, pred_play) = cost(m, pred_play, c)

function cost(m::Chain, pred_play, c::DeepCosts)
    res = 0
    res += c.γ*sum(norm, params(m))
    res += c.exact/Flux.crossentropy(pred_play,pred_play)
    sim_dist = my_softmax(m.layers[end].v)
    res += sum([c.sim*(i-1)*sim_dist[i] for i in 1:length(sim_dist)])
    return res
end


mutable struct DeepCostSpace
    γ_min::Float64
    γ_max::Float64
    exact_min::Float64
    exact_max::Float64
    sim_min::Float64
    sim_max::Float64
end

DeepCostSpace(γ::Tuple{Float64, Float64}, exact::Tuple{Float64, Float64}, sim::Tuple{Float64, Float64}) = DeepCostSpace(γ[1], γ[2], exact[1], exact[2], sim[1], sim[2])


function rand(cs::DeepCostSpace)
    γ = scale(rand(), cs.γ_min, cs.γ_max)
    exact = scale(rand(), cs.exact_min, cs.exact_max)
    sim = scale(rand(), cs.sim_min, cs.sim_max)
    DeepCosts(γ,exact,sim)
end

function rand(cs::DeepCostSpace, n::Int64)
    [rand(cs) for i in 1:n]
end

opt = ADAM(0.001, (0.9, 0.999))

function optimize_model(base_model::Chain, data, costs::DeepCosts; n_iter=20)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
    loss(x::Game, y) = begin
        pred_play = model(x)
        -expected_payoff(pred_play, empirical_play, x) + costs(model, pred_play)
    end
    ps = Flux.params(model)
    for i in 1:n_iter
        Flux.train!(loss, ps, data, opt)
    end
    model
end

function fit_model(base_model::Chain, data, costs::DeepCosts; n_iter=20)
    model = deepcopy(base_model)
    loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
    loss(x::Game, y) = Flux.crossentropy(model(x), y) + costs.γ*sum(norm, params(model))
    ps = Flux.params(model)
    for i in 1:n_iter
        Flux.train!(loss, ps, data, opt)
    end
    model
end


####################################################################
##%% Rule Learning
####################################################################

function fit_model(base_model::RuleLearning, data, idx, costs::Costs; n_iter=1)
    model = deepcopy(base_model)
    model.costs = costs
    for i in 1:n_iter
        model = fit_βs_and_prior(model, data, idx)
        model = optimize_rule_lambdas(model, data, idx)
    end
    model
end


####################################################################
##%% Multiple dispatch with learning
####################################################################
function fit_model(model, data, idx, costs)
    fit_model(model, data[idx], costs)
end

function optimize_model(model, data, idx, costs)
    optimize_model(model, data[idx], costs)
end
