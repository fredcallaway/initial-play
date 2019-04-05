using Flux
using Flux: @epochs
using Flux.Tracker
using Flux.Tracker: grad, update!
using LinearAlgebra: norm
using StatsBase: entropy
using JSON
using CSV
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
mh = MetaHeuristic([JointMax(3.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [30., 30., 30., 0., 0., 0., 0.]);
h_dists = [h_distribution(mh, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)

data =  [(g, play_distribution(mh, g)) for g in games];
test_data = [(g, play_distribution(mh, g)) for g in test_games];

####################################################
#%% Set up Deep heuristic and loss functions
####################################################

# model = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3));
model = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense_full(50,50, sigmoid), Game_Dense(50,30), Game_Soft(30),  Last(1));

loss(x::Game, y) =Flux.crossentropy(model(x), y) + 0.001*sum(norm, params(model))
loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.01)./1.03, (y .+ 0.01)./1.03) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)


@show rand_loss(data)
@show loss(data)
@show loss_no_norm(data)
@show min_loss(data);

####################################################
#%% Collect parameters and run heuristic
####################################################


ps = Flux.params(model)
opt = ADAM(0.01, (0.9, 0.999))
evalcb() = @show(rand_loss(data), loss_no_norm(data), min_loss(data), rand_loss(test_data), loss_no_norm(test_data), min_loss(test_data))
println(5) # This is just so the automatic printing from Atom is nice
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
model_0 = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense_full(50,50, sigmoid), Game_Dense_full(50,20), Game_Soft(20), Last(1))
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
@epochs 30 Flux.train!(loss, ps, train_exp_data, opt, cb = Flux.throttle(evalcb,5))

####################################################
#%% Add action response layers
####################################################
model_action = Chain(model_0.layers[1:(length(model_0.layers)-1)]..., Action_Response(1), Last(2))
loss(x::Game, y) = Flux.crossentropy(model_action(x), y) + 0.001*sum(norm, params(model_action))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_action(x), y)

loss(exp_data)
#%% Estimate with action response layers
ps = Flux.params(model_action)
opt = ADAM(0.001, (0.9, 0.999))
5
evalcb() = @show(rand_loss(train_exp_data), loss_no_norm(train_exp_data), min_loss(train_exp_data), rand_loss(test_exp_data), loss_no_norm(test_exp_data), min_loss(test_exp_data))
@epochs 10 Flux.train!(loss, ps, train_exp_data, opt, cb = Flux.throttle(evalcb,5))


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


####################################################
#%% Estimate on Pilot data
####################################################
function json_to_game(s)
    a = JSON.parse(s)
    row = [convert(Float64, a[i][j][1]) for i in 1:length(a), j in 1:length(a[1])]
    col = [convert(Float64, a[i][j][2]) for i in 1:length(a), j in 1:length(a[1])]
    row_g = Game(row, col)
end

particapant_df = CSV.read("Gustav_pilot/dataframes/participant_df.csv")
individal_choices_df = CSV.read("Gustav_pilot/dataframes/individal_choices_df.csv")
positive_games_df = CSV.read("Gustav_pilot/dataframes/positive_games_df.csv")

comparison_idx = [31, 37, 41, 44, 49]
later_com = 50 .+ comparison_idx
comparison_idx = [comparison_idx..., later_com...]

pilot_data = [(json_to_game(first(positive_games_df[positive_games_df.round .== i, :row])), convert(Vector{Float64}, JSON.parse(first(positive_games_df[positive_games_df.round .== i, :row_play])))) for i in 1:50];
println(pilot_data[1])
append!(pilot_data ,[(transpose(json_to_game(first(positive_games_df[positive_games_df.round .== i, :col]))), convert(Vector{Float64}, JSON.parse(first(positive_games_df[positive_games_df.round .== i, :col_play])))) for i in 1:50]);

pilot_games = [d[1] for d in pilot_data];
pilot_row_plays = [d[2] for d in pilot_data];
pilot_col_plays = [pilot_row_plays[51:100]..., pilot_row_plays[1:50]...]


# pilot_n_train = 70
# pilot_train_idxs = sample(1:length(pilot_games), pilot_n_train; replace=false)
# pilot_test_idxs = setdiff(1:length(pilot_games), pilot_train_idxs)
pilot_train_idxs = setdiff(1:length(pilot_games), comparison_idx)
pilot_test_idxs = comparison_idx
sort!(pilot_train_idxs)
sort!(pilot_test_idxs)
pilot_train_games = pilot_games[pilot_train_idxs];
pilot_test_games = pilot_games[pilot_test_idxs];
pilot_train_row = pilot_row_plays[pilot_train_idxs]
pilot_test_row = pilot_row_plays[pilot_test_idxs]
pilot_train_data = pilot_data[pilot_train_idxs];
pilot_test_data = pilot_data[pilot_test_idxs];


####################################################
#%% QLK Pilot
####################################################
actual_h = CacheHeuristic(pilot_games, pilot_row_plays)
pilot_opp_h = CacheHeuristic(pilot_games, pilot_col_plays)
# actual_h_test = CacheHeuristic(pilot_test_games, [play_distribution(mh, g) for g in pilot_test_games])
qlk_h = QLK(0.07, 0.64, 2.3)
best_fit_qlk = fit_h!(qlk_h, pilot_games, actual_h)
# best_fit_qlk = fit_h!(qlk_h, pilot_games[comparison_idx], actual_h)
# prediction_loss(qlk_h, pilot_games[comparison_idx], actual_h)
prediction_loss(qlk_h, pilot_games, actual_h)
best_fit_qlk = fit_h!(qlk_h, pilot_train_games, actual_h)
@show prediction_loss(qlk_h, pilot_train_games, actual_h);
@show prediction_loss(qlk_h, pilot_test_games, actual_h);

@show prediction_loss(qlk_h, pilot_games[comparison_idx], actual_h)

####################################################
#%% QLK Pilot
####################################################
qch_h = QCH(0.07, 0.64, 1.5, 1.7, 1.9)
best_fit_qch = fit_h!(qch_h, pilot_games, actual_h)
# best_fit_qch = fit_h!(qch_h, pilot_games[comparison_idx], actual_h)
# prediction_loss(qch_h, pilot_games[comparison_idx], actual_h)
prediction_loss(qch_h, pilot_games, actual_h)
best_fit_qch = fit_h!(qch_h, pilot_train_games, actual_h)
@show prediction_loss(qch_h, pilot_train_games, actual_h);
@show prediction_loss(qch_h, pilot_test_games, actual_h);

@show prediction_loss(qch_h, pilot_games[comparison_idx], actual_h)


####################################################
#%% MetaHeuristic Pilot
####################################################
mh = MetaHeuristic([JointMax(1.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);

costs = Costs(0.1, 0.25, 0.6, 4.5)
opt_mh = mh
opt_mh = fit_prior!(opt_mh, pilot_train_games, actual_h, pilot_opp_h, costs)
opt_mh = fit_h!(mh, pilot_train_games, actual_h, pilot_opp_h, costs; init_x = get_parameters(mh))
opt_mh = opt_prior!(opt_mh, pilot_train_games, pilot_opp_h, costs)
opt_mh = optimize_h!(opt_mh, pilot_train_games, pilot_opp_h, costs)

prediction_loss(opt_mh, pilot_games, actual_h)
prediction_loss(opt_mh, pilot_train_games, actual_h)
prediction_loss(opt_mh, pilot_test_games, actual_h)
prediction_loss(opt_mh, pilot_games[comparison_idx], actual_h)

h_dists = [h_distribution(opt_mh, g, pilot_opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)

####################################################
#%% Pilot Setup and run Deep Heuristic without action response
####################################################
γ = 0.001 # Overfitting penalty
# model_0 = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
model_0 = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Last(1))
# model_0 = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

loss(x::Game, y) = Flux.crossentropy(model_0(x), y) + γ*sum(norm, params(model_0))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_0(x), y)

#%% estimate
ps = Flux.params(model_0)
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(rand_loss(pilot_train_data), loss_no_norm(pilot_train_data), min_loss(pilot_train_data), rand_loss(pilot_test_data), loss_no_norm(pilot_test_data), min_loss(pilot_test_data))
println("clear print are")
5
@epochs 20 Flux.train!(loss, ps, pilot_train_data, opt, cb = Flux.throttle(evalcb,5))

####################################################
#%% Add action response layers
####################################################
model_action = deepcopy(Chain(model_0.layers[1:(length(model_0.layers)-1)]..., Action_Response(1), Last(2)))
loss(x::Game, y) = Flux.crossentropy(model_action(x), y) + γ*sum(norm, params(model_action))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_action(x), y)

loss_no_norm(pilot_data[comparison_idx])
loss_no_norm(pilot_data)
#%% Estimate with action response layers
ps = Flux.params(model_action)
opt = ADAM(0.0001, (0.9, 0.999))
5
evalcb() = @show(rand_loss(pilot_train_data), loss_no_norm(pilot_train_data), min_loss(pilot_train_data), rand_loss(pilot_test_data), loss_no_norm(pilot_test_data), min_loss(pilot_test_data))
println("apa")
4
@epochs 100 Flux.train!(loss, ps, pilot_train_data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% Inspect perfomance on individual games
####################################################
m_correct = 0
q_correct = 0
miss_idx = []

for (i, (game, play)) in enumerate(pilot_data)
    # println(game_names[i])
    println(i)
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

miss_games = pilot_games[miss_idx];
miss_plays = pilot_row_plays[miss_idx];
best_fit_qlk = fit_h!(qlk_h, miss_games, actual_h)
prediction_loss(qlk_h, miss_games, actual_h)

for i in miss_idx
    println(i)
    println(pilot_data[i][1])
    println(pilot_data[i][2])
    println(model_action(pilot_data[i][1]))
end


####################################################
#%% Pilot optimal Depp heuristic
####################################################
γ = 0.001 # Overfitting penalty
sim_cost = 0.1
exact_cost = 0.8
# model_0 = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
model = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense(50,50), Game_Soft(50), Action_Response(1), Last(2))
# model_0 = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

# loss(x::Game, y) = Flux.crossentropy(model_0(x), y) + γ*sum(norm, params(model_0))
loss(x::Game, y) = begin
    pred_play = model(x)
    -expected_payoff(pred_play, pilot_opp_h, x) + γ*sum(norm, params(model)) + sim_cost*my_softmax(model.layers[end].v)[2] + exact_cost/Flux.crossentropy(pred_play,pred_play)
end
loss_no_cost(x::Game, y) = -expected_payoff(model(x), pilot_opp_h, x)
loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
loss_no_cost(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_cost(g,y) for (g,y) in data])/length(data)
# loss(pilot_data[1][1], pilot_data[1][2])



loss(pilot_data)
loss_no_cost(pilot_data)
loss_no_norm(pilot_data)

gid = 35
pilot_data[gid][2]
model_action(pilot_data[gid][1])
model(pilot_data[gid][1])
Flux.crossentropy(model(pilot_data[20][1]), model(pilot_data[20][1]))
Flux.crossentropy(ones(3)/3, ones(3)/3)

# softmax(model.layers[end].v)
#%% estimate
ps = Flux.params(model)
opt = ADAM(0.001, (0.9, 0.999))
# evalcb() = @show(loss(pilot_data), rand_loss(pilot_train_data), loss_no_norm(pilot_train_data), min_loss(pilot_train_data), rand_loss(pilot_test_data), loss_no_norm(pilot_test_data), min_loss(pilot_test_data))
evalcb() = @show(loss(pilot_data), loss_no_cost(pilot_data), loss_no_norm(pilot_data), min_loss(pilot_data), loss_no_norm(pilot_test_data))
println("clear print are")
5
@epochs 10 Flux.train!(loss, ps, pilot_data, opt, cb = Flux.throttle(evalcb,5))
