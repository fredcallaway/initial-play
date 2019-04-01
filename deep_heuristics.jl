using Flux
using Flux: @epochs
using Flux.Tracker
using Flux.Tracker: grad, update!
using LinearAlgebra: norm
using JSON
include("Heuristics.jl")
include("DeepLayers.jl")

####################################################
#%% Generate some data to test the Deep Heuristic on
####################################################
opp_h = QLK(0.1, 0.6, 3.)

costs = Costs(0.03, 0.1, 0.3, 5.)
ρ = 0.9
games = [Game(3, ρ) for i in 1:100];
test_games = [Game(3, ρ) for i in 1:100];
mh = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [30., 30., 0., 0., 0., 0., 30.]);
h_dists = [h_distribution(mh, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)

data =  [(g, play_distribution(mh, g)) for g in games];
test_data = [(g, play_distribution(mh, g)) for g in test_games];

####################################################
#%% Set up Deep heuristic and loss functions
####################################################

model = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3));

loss(x::Game, y) =Flux.crossentropy(model(x), y) + 0.001*sum(norm, params(model))
loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.01)./1.03, (y .+ 0.01)./1.03) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)


min_loss(data)
rand_loss(data)
loss(data)
loss_no_norm(data)

####################################################
#%% Collect parameters and run heuristic
####################################################


ps = Flux.params(model)
opt = ADAM(0.01, (0.9, 0.999))
evalcb() = @show(rand_loss(data), loss_no_norm(data), min_loss(data), rand_loss(test_data), loss_no_norm(test_data), min_loss(test_data))
@epochs 1 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% Compare with perfomance of QLK
####################################################

actual_h = CacheHeuristic(games, [play_distribution(mh, g) for g in games])
actual_h_test = CacheHeuristic(test_games, [play_distribution(mh, g) for g in test_games])
qlk_h = QLK(0.07, 0.64, 2.3)
# best_fit_qlk = fit_h!(qlk_h, exp_games, actual_h; loss_f = mean_square)
best_fit_qlk = fit_h!(qlk_h, games, actual_h)
@show rand_loss(data)
@show prediction_loss(qlk_h, games, actual_h)
@show min_loss(data)
@show rand_loss(test_data)
@show prediction_loss(qlk_h, test_games, actual_h_test)
@show min_loss(test_data);

####################################################
#%% Load real data
####################################################
data_name_list = open("data/dataset_list.json") do f
        JSON.parse(read(f, String))
end

game_names = open("data/game_names_list.json") do f
    JSON.parse(read(f, String))
end

data_sets = unique(data_name_list)
exp_games = convert(Vector{Game}, games_from_json("data/games.json"))
exp_row_plays = plays_vec_from_json("data/rows_played.json")
exp_col_plays = plays_vec_from_json("data/cols_played.json")


n_train = 100
train_idxs = sample(1:length(exp_games), n_train; replace=false)
test_idxs = setdiff(1:length(exp_games), train_idxs)
sort!(train_idxs)
sort!(test_idxs)
train_exp_games = exp_games[train_idxs]
test_exp_games = exp_games[test_idxs]
train_row = exp_row_plays[train_idxs]
test_row = exp_row_plays[test_idxs];

####################################################
#%% Performance of QLK
####################################################
actual_h = CacheHeuristic(exp_games, exp_row_plays)
# actual_h_test = CacheHeuristic(test_exp_games, [play_distribution(mh, g) for g in test_exp_games])
qlk_h = QLK(0.07, 0.64, 2.3)
best_fit_qlk = fit_h!(qlk_h, exp_games, actual_h)
prediction_loss(qlk_h, exp_games, actual_h)
best_fit_qlk = fit_h!(qlk_h, train_exp_games, actual_h)
@show prediction_loss(qlk_h, train_exp_games, actual_h)
@show prediction_loss(qlk_h, test_exp_games, actual_h);


####################################################
#%% Setup and run Deep Heuristic without action response
####################################################
model_0 = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense_full(50,50, sigmoid), Game_Dense(50,40), Game_Soft(40), Last(1))
# model_0 = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))
exp_data = [(exp_games[i],exp_row_plays[i]) for i in 1:length(exp_games)];
train_exp_data = exp_data[train_idxs];
test_exp_data = exp_data[test_idxs];

loss(x::Game, y) = Flux.crossentropy(model_0(x), y) + 0.001*sum(norm, params(model_0))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_0(x), y)

#%% estimate
ps = Flux.params(model_0)
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(rand_loss(train_exp_data), loss_no_norm(train_exp_data), min_loss(train_exp_data), rand_loss(test_exp_data), loss_no_norm(test_exp_data), min_loss(test_exp_data))
@epochs 1 Flux.train!(loss, ps, train_exp_data, opt, cb = Flux.throttle(evalcb,5))

####################################################
#%% Add action response layers
####################################################
model_action = Chain(model_0.layers[1:(length(model_0.layers)-1)]..., Action_Response(1), Action_Response(2), Last(3))
loss(x::Game, y) = Flux.crossentropy(model_action(x), y) + 0.001*sum(norm, params(model_action))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_action(x), y)

loss(exp_data)
#%% Estimate with action response layers
ps = Flux.params(model_action)
opt = ADAM(0.0001, (0.9, 0.999))
cb() = @show(rand_loss(train_exp_data), loss_no_norm(train_exp_data), min_loss(train_exp_data), rand_loss(test_exp_data), loss_no_norm(test_exp_data), min_loss(test_exp_data))
@epochs 1 Flux.train!(loss, ps, train_exp_data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% Inspect perfomance on individual games
####################################################
m_correct = 0
q_correct = 0
miss_idx = []
for (i, (game, play)) in enumerate(exp_data)
    println(game_names[i])
    println(game)
    println(play)
    q_play = play_distribution(qlk_h, game)
    m_play = model_action(game)
    m_correct += findmax(m_play)[2] == findmax(play)[2]
    q_correct += findmax(q_play)[2] == findmax(play)[2]
    if findmax(m_play)[2] != findmax(play)[2]
        append!(miss_idx, i)
    end
    println(play_distribution(qlk_h, game))
    println(model_action(game))
    println("----------------")
end
println(m_correct)
println(q_correct)

####################################################
#%% Estimate on the misspecified data
####################################################
# Compare QLK
miss_games = exp_games[miss_idx]
miss_plays = exp_row_plays[miss_idx]
best_fit_qlk = fit_h!(qlk_h, miss_games, actual_h)
prediction_loss(qlk_h, miss_games, actual_h)


#%% Deep heuristic on train data
miss_data = exp_data[miss_idx]
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(rand_loss(miss_data), loss_no_norm(miss_data), min_loss(miss_data), loss_no_norm(data))
5
@epochs 1 Flux.train!(loss, ps, miss_data, opt, cb = Flux.throttle(evalcb,5))
