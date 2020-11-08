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


# function fit_model(base_model::MetaHeuristic, data, costs::Costs; n_iter=5)
#     games, plays = invert(data)
#     empirical_play = CacheHeuristic(games, plays);
#     model = deepcopy(base_model)
#     for i in 1:n_iter
#         fit_prior!(model, games, empirical_play, empirical_play, costs)
#         fit_h!(model, games, empirical_play, empirical_play, costs)
#     end
#     model
# end

function fit_model(base_model::SoftmaxHeuristic, data, costs::Costs; n_iter=5)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    init_x = [get_parameters(model)...]
    set_as!(model, empirical_play, games, costs)
    function loss_wrap(x)
        set_parameters!(model, x)
        update_h_dist!(model)
        prediction_loss(model, games, empirical_play)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    set_parameters!(model, x[1:end])
    set_as!(model, empirical_play, games, costs)
    update_h_dist!(model)
    model
end

function fit_model(base_model::MetaHeuristic, data, costs::Costs; n_iter=5)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    init_x = [get_parameters(model)..., model.prior...]
    len_params = length(get_parameters(model))
    function loss_wrap(x)
        set_parameters!(model, x[1:len_params])
        model.prior = x[len_params+1:end]
        prediction_loss(model, games, empirical_play, empirical_play, costs)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    set_parameters!(model, x[1:len_params])
    model.prior = x[len_params+1:end]
    model
end

function fit_model_no_RI(base_model::MetaHeuristic, data)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays)
    model = deepcopy(base_model)
    init_x = [get_parameters(model)..., model.prior...]
    len_params = length(get_parameters(model))
    function loss_wrap(x)
        set_parameters!(model, x[1:len_params])
        model.prior = x[len_params+1:end]
        prediction_loss(model, games, empirical_play)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    set_parameters!(model, x[1:len_params])
    model.prior = x[len_params+1:end]
    model
end



# function optimize_model(base_model::MetaHeuristic, data, costs::Costs; n_iter=5)
#     games, plays = invert(data)
#     empirical_play = CacheHeuristic(games, plays);
#     model = deepcopy(base_model)
#     for i in 1:n_iter
#         optimize_h!(model, games, empirical_play, costs)
#         opt_prior!(model, games, empirical_play, costs)
#     end
#     model
# end

function optimize_model(base_model::SoftmaxHeuristic, data, costs::Costs)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    model.λ = costs.m_λ
    for h in model.h_list
        optimize_h!(h, games, empirical_play, costs)
    end
    set_as!(model, empirical_play, games, costs)
    update_h_dist!(model)
    model
end

function optimize_model(base_model::MetaHeuristic, data, costs::Costs; n_iter=5)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    model = deepcopy(base_model)
    init_x = [get_parameters(model)..., model.prior...]
    len_params = length(get_parameters(model))
    function loss_wrap(x)
        set_parameters!(model, x[1:len_params])
        model.prior = x[len_params+1:end]
        -perf(model, games, empirical_play, costs)
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    set_parameters!(model, x[1:len_params])
    model.prior = x[len_params+1:end]
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

DeepCosts(;γ, exact, sim) = DeepCosts(γ, exact, sim)

(c::DeepCosts)(m::Chain, pred_play) = cost(m, pred_play, c)

function cost(m::Chain, pred_play, c::DeepCosts)
    res = 0
    res += c.exact/Flux.crossentropy(pred_play,pred_play)
    sim_dist = softmax(m.layers[end].v)
    for i in 1:length(sim_dist)
        res += c.sim*(i-1)*sim_dist[i]
    end
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

function optimize_model(base_model::Chain, data, costs::DeepCosts; n_iter=10)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    feats = gen_feats.(games)
    dat = collect(zip(feats, games, plays))
    model = deepcopy(base_model)

    loss(data::Array{Tuple{Tuple{Array{Real,2},Array{Real,2},Array{Array{Float64,1},2}},Game,Array{Float64,1}},1}) = mean(loss.(data))
    loss(d::Tuple{Tuple{Array{Real,2},Array{Real,2},Array{Array{Float64,1},2}},Game,Array{Float64,1}}) = loss(d...)
    loss(x::Features, g::Game, y) = begin
        pred_play = model(x)
        -expected_payoff(pred_play, empirical_play, g) + costs(model, pred_play)
    end

    ps = Flux.params(model)
    for i in 1:n_iter
        Flux.train!(loss, ps, dat, opt)
    end
    model
end


function fit_model(base_model::Chain, data, costs::DeepCosts; n_iter=10)
    games, play = invert(data)
    feats = gen_feats.(games)
    dat = collect(zip(feats, play))
    model = deepcopy(base_model)
    loss(data::Array{Tuple{Features,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
    loss(x::Features, y) = Flux.crossentropy(model(x), y)
    ps = Flux.params(model)
    for i in 1:n_iter
        Flux.train!(loss, ps, dat, opt)
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


# %% ==================== Loading Data ====================
@everywhere Data = Array{Tuple{Game,Array{Float64,1}},1}
parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))

function load_data(file)::Data
    file
    df = CSV.read(file);
    row_plays = Data()
    col_plays = Data()
    for row in eachrow(df)
        row_game = json_to_game(row.row_game)
        row_game.row[1,2] += rand()*1e-7 # This can't be a constant number if we want to
        row_game.col[1,2] += rand()*1e-7 # separate behavior in comparison games in different treatments.
        row_play_dist = parse_play(row.row_play)
        col_game = transpose(row_game)
        col_play_dist = parse_play(row.col_play)
        push!(row_plays, (row_game, row_play_dist))
        push!(col_plays, (col_game, col_play_dist))
    end
    append!(row_plays, col_plays)
end

function load_treatment_data(treatment)
    files = glob("data/processed/$treatment/*_play_distributions.csv")
    data = vcat(map(load_data, files)...)
end

# %% ==================== Loss functions ====================
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)

loss_min(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)

function prediction_loss(model::MetaHeuristic, in_data::Data, idx, costs)
    data = in_data[idx]
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play, empirical_play, costs)
end

function prediction_loss_no_RI(model::MetaHeuristic, in_data::Data, idx, costs)
    data = in_data[idx]
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play)
end

function prediction_loss(model::Heuristic, in_data::Data, idx, costs)
    data = in_data[idx]
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play)
end

function prediction_loss(model::Chain, in_data::Data, idx, costs)
    data = in_data[idx]
    loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
    loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
    loss_no_norm(data)
end

function prediction_loss(model::RuleLearning, in_data::Data, idx, costs)
    rule_loss_idx(model, in_data, idx)
end


# %% ==================== Train test subsetting ====================
function comparison_indices(data)
    comparison_idx = [31, 38, 42, 49]
    comparison_idx = vcat([comparison_idx .+ 50*i for i in 0:40]...)
    # later_com = 50 .+ comparison_idx
    # comparison_idx = [comparison_idx..., later_com...]
    comparison_idx = filter(x -> x <= length(data), comparison_idx)
    sort(comparison_idx)
end

function early_late_indices(data; n =30)
    train_idx = filter(x -> (x-1) % 50 < n, 1:length(data))
    test_idx = setdiff(1:length(data), train_idx)
    test_idx = setdiff(test_idx, comparison_indices(data))
    train_idx = setdiff(train_idx, comparison_indices(data))
    train_idx, test_idx
end

function leave_one_pop_out_indices(data, test_pop)
    test_idx = collect(1:100) .+ (test_pop-1)*100
    train_idx = setdiff(1:length(data), test_idx)
    test_idx = setdiff(test_idx, comparison_indices(data))
    train_idx = setdiff(train_idx, comparison_indices(data))
    train_idx, test_idx
end
