using Flux
using JSON
using CSV
using DataFrames
using SplitApplyCombine
using Random
using Glob
using Distributed
include("Heuristics.jl")

# %% ====================  ====================
Data = Array{Tuple{Game,Array{Float64,1}},1}
parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))

function load_data(file)::Data
    df = CSV.read(file);
    row_plays = map(eachrow(df)) do x
        game = json_to_game(x.row_game)
        game.row[1, 2] += 1e-5
        game.col[1, 2] -= 1e-5
        # break_symmetry!(game)
        play_dist = parse_play(x.row_play)
        (game, play_dist)
    end
    col_plays = map(eachrow(df)) do x
        game = json_to_game(x.col_game)
        game.row[2, 1] -= 1e-5
        game.col[2, 1] += 1e-5
        # break_symmetry!(game)
        play_dist = parse_play(x.col_play)
        (game, play_dist)
    end
    append!(row_plays, col_plays)
end


# %% ==================== Loss functions ====================
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)

loss_min(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)

function prediction_loss(model, data::Data)
    games, plays = invert(data)
    empirical_play = CacheHeuristic(games, plays);
    prediction_loss(model, games, empirical_play)
end


# %% ==================== Train test subsetting ====================
function comparison_indices()
    indices = ...
    indices
end

function early_late_indices(data)
    # first 30 go to train, last 20 to test
    indices = ...
    indices, indices
end

function optimize!(model::MetaHeuristic, data::Data)
    # fits costs assuming optimality
    # return model and costs
end

function fit!(model::MetaHeuristic, data::Data)
    # fits costs and parameters
    # return model and costs
end

# TODO implement these for all models
# No optimize for RuleLearning

function run(model, data, train_idx, test_idx, train!)
    # train! ∈ {optimize!, fit!}
    # maybe do comparison prediction here as well?
    model, costs = train!(deepcopy(model), data[train_idx])
    (
        train_loss = prediction_loss(model, data[train_idx]),
        test_loss = prediction_loss(model, data[test_idx]),
        comparison_loss = prediction_loss(model, data[comparison_indices(data)]),
        model = model,
        costs = costs,
    )

end

#= Analysis Plan

1. Implement Train 30 Test 20
2. Implement LOOCV at the population level
3. Fit and optimize MetaHeuritic
    - Fit costs when fitting parameters
4. Fit and optimize Deep Heuristic
5. Fit RuleLearning

=#

# %% ====================  ====================
treatment = "positive"
files = glob("data/processed/$treatment/*_play_distributions.csv")
data = vcat(map(load_data, files)...)
typeof(data)


# %% ====================  ====================

file = "data/processed/positive$(session)_play_distributions.csv"
load_data(file)
load_data("pilot_pos") |> typeof
