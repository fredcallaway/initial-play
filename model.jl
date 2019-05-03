# using Flux
# using Flux: @epochs
# using Flux.Tracker
# using Flux.Tracker: grad, update!
# using LinearAlgebra: norm
# using StatsBase: entropy
using JSON
using CSV
using DataFrames
# using DataFramesMeta
using SplitApplyCombine
using Random
include("Heuristics.jl")
# include("DeepLayers.jl")


function json_to_game(s)
    a = JSON.parse(s)
    row = [convert(Float64, a[i][j][1]) for i in 1:length(a), j in 1:length(a[1])]
    col = [convert(Float64, a[i][j][2]) for i in 1:length(a), j in 1:length(a[1])]
    row_g = Game(row, col)
end
###########################################################################
# %% Load the data
###########################################################################

# particapant_df = CSV.read("pilot/dataframes/participant_df.csv")
# individal_choices_df = CSV.read("pilot/dataframes/individal_choices_df.csv")
competing_games_df = CSV.read("data/processed/e3jydlve_play_distributions.csv");
common_games_df = CSV.read("data/processed/nlesp5ss_play_distributions.csv");


# %% General loss functions
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)

loss_min(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)

costs = Costs(0.1682716344671231, 0.18753820682175806, 0.06078936090685586, 1.0690653407809032)


###########################################################################
# %% common Treatment: Load the data
###########################################################################
comparison_idx = [31, 34, 38, 41, 44, 50]

later_com = 50 .+ comparison_idx
comparison_idx = [comparison_idx; later_com]

# function break_symmetry!(g::Game)
#     if transpose(g.row) == g.col
#         g.row[1] += 0.0001
#     end
# end

parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))
function games_and_plays(df)
    row_plays = map(eachrow(df)) do x
        game = json_to_game(x.row_game)
        # break_symmetry!(game)
        play_dist = parse_play(x.row_play)
        (game, play_dist)
    end
    col_plays = map(eachrow(df)) do x
        game = json_to_game(x.col_game)
        # break_symmetry!(game)
        play_dist = parse_play(x.col_play)
        (game, play_dist)
    end
    result = append!(row_plays, col_plays)
    # Break the symetry for the one symetric game.
    result[41][1].row[2] += 0.0001
    result[41][1].col[2] -= 0.0001
    result[91] = (transpose(result[41][1]), result[91][2])
    result
end

const common_data = games_and_plays(common_games_df)
const competing_data = games_and_plays(competing_games_df)

# %% ====================  ====================

function cross_validate(train, test, data; k=5, seed=1, parallel=false)
    n = length(data)
    n2 = fld(n,2)
    indices = shuffle(MersenneTwister(seed), 1:n2)
    chunks = Iterators.partition(indices, div(n2,k)) |> collect
    for ch in chunks
        col_indices = ch .+ 50
        push!(ch, col_indices...)
    end

    mymap = parallel ? pmap : map
    mymap(1:k) do i
        test_indices = chunks[i]
        train_indices = setdiff(1:n, test_indices)
        model = train(data[train_indices])
        test(model, data[test_indices])
    end
end

function make_fit(base_model::QCH)
    data -> begin
        games, plays = invert(data)
        empirical_play = CacheHeuristic(games, plays);
        fit_h!(deepcopy(base_model), games, empirical_play)
    end
end

function make_optimize(base_model::QCH, costs=costs)
    data -> begin
        games, plays = invert(data)
        empirical_play = CacheHeuristic(games, plays);
        optimize_h!(deepcopy(base_model), games, empirical_play, costs)
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


function make_fit(base_model::MetaHeuristic; n_iter=5)
    @assert false # FIXME: need to pass costs!
    data -> begin
        games, plays = invert(data)
        empirical_play = CacheHeuristic(games, plays);
        model = deepcopy(base_model)
        for i in 1:n_iter
            fit_prior!(model, games, empirical_play, empirical_play, costs)
            fit_h!(model, games, empirical_play, empirical_play, costs)
        end
        model
    end
end

function make_optimize(base_model::MetaHeuristic, costs=costs; n_iter=5)
    data -> begin
        games, plays = invert(data)
        empirical_play = CacheHeuristic(games, plays);
        model = deepcopy(base_model)
        for i in 1:n_iter
            optimize_h!(model, games, empirical_play, costs)
            opt_prior!(model, games, empirical_play, costs)
        end
        model
    end
end

const train_indices, test_indices = let
    train_indices = collect(1:30)
    push!(train_indices, (train_indices .+ 50)...)
    test_indices = setdiff(1:100, train_indices)
    train_indices, test_indices
end


