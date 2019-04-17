using Flux
using Flux: @epochs
using Flux.Tracker
using Flux.Tracker: grad, update!
using LinearAlgebra: norm
using StatsBase: entropy
using JSON
using CSV
using DataFrames
using DataFramesMeta
include("Heuristics.jl")
include("DeepLayers.jl")

function json_to_game(s)
    a = JSON.parse(s)
    row = [convert(Float64, a[i][j][1]) for i in 1:length(a), j in 1:length(a[1])]
    col = [convert(Float64, a[i][j][2]) for i in 1:length(a), j in 1:length(a[1])]
    row_g = Game(row, col)
end
###########################################################################
#%% Load the data
###########################################################################

particapant_df = CSV.read("pilot/dataframes/participant_df.csv")
individal_choices_df = CSV.read("pilot/dataframes/individal_choices_df.csv")
common_games_df = CSV.read("pilot/dataframes/positive_games_df.csv")
competing_games_df = CSV.read("pilot/dataframes/negative_games_df.csv")


comparison_idx = [31, 37, 41, 44, 49]
#%% Look at comparison games

for i in comparison_idx
    game = json_to_game(first(common_games_df[common_games_df.round .== i, :row]))
    println(game)
    pos_row_play = pos_dfrow = first(common_games_df[common_games_df.round .== i, :row_play])
    pos_col_play = pos_dfrow = first(common_games_df[common_games_df.round .== i, :col_play])
    println("pos row: ", pos_row_play)
    println("pos col: ", pos_col_play)
    neg_row_play = neg_dfrow = first(competing_games_df[competing_games_df.round .== i, :row_play])
    neg_col_play = neg_dfrow = first(competing_games_df[competing_games_df.round .== i, :col_play])
    println("neg row: ", neg_row_play)
    println("neg col: ", neg_col_play)
    println("------------------------------------------------------")
end


#%% General loss functions and parameters
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)

loss_min(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.001)./1.003, (y .+ 0.001)./1.003) : Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_min(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)

costs = Costs(0.2, 0.2, 0.1, 1.5)
n_epochs = 100
γ = 0.003 # Overfitting penalty
sim_cost = 0.2
exact_cost = 0.2
opt = ADAM(0.001, (0.9, 0.999))

###########################################################################
#%% common Treatment: Load the data
###########################################################################
comparison_idx = [31, 37, 41, 44, 49]
later_com = 50 .+ comparison_idx
comparison_idx = [comparison_idx..., later_com...]

pilot_pos_data = [(json_to_game(first(common_games_df[common_games_df.round .== i, :row])), convert(Vector{Float64}, JSON.parse(first(common_games_df[common_games_df.round .== i, :row_play])))) for i in 1:50];
append!(pilot_pos_data ,[(json_to_game(first(common_games_df[common_games_df.round .== i, :col])), convert(Vector{Float64}, JSON.parse(first(common_games_df[common_games_df.round .== i, :col_play])))) for i in 1:50]);

# By adding a minor random noise to the only symmetric game we can distinguish the game from the row and columns perspective
pilot_pos_data[41] = (Game(pilot_pos_data[41][1].row .+ rand(3,3)*0.000001, pilot_pos_data[41][1].col .+ rand(3,3)*0.000001), pilot_pos_data[41][2])
pilot_pos_data[91] = (transpose(pilot_pos_data[41][1]), pilot_pos_data[91][2])


pilot_pos_games = [d[1] for d in pilot_pos_data];
pilot_pos_row_plays = [d[2] for d in pilot_pos_data];
pilot_pos_col_plays = [pilot_pos_row_plays[51:100]..., pilot_pos_row_plays[1:50]...]

pilot_pos_n_train = 70;
pilot_pos_train_idxs = sample(1:length(pilot_pos_games), pilot_pos_n_train; replace=false);
pilot_pos_test_idxs = setdiff(1:length(pilot_pos_games), pilot_pos_train_idxs);
# pilot_pos_train_idxs = setdiff(1:length(pilot_pos_games), comparison_idx)
# pilot_pos_test_idxs = comparison_idx
sort!(pilot_pos_train_idxs)
sort!(pilot_pos_test_idxs)
pilot_pos_train_games = pilot_pos_games[pilot_pos_train_idxs];
pilot_pos_test_games = pilot_pos_games[pilot_pos_test_idxs];
pilot_pos_train_row = pilot_pos_row_plays[pilot_pos_train_idxs];
pilot_pos_test_row = pilot_pos_row_plays[pilot_pos_test_idxs];
pilot_pos_train_data = pilot_pos_data[pilot_pos_train_idxs];
pilot_pos_test_data = pilot_pos_data[pilot_pos_test_idxs];

####################################################
#%% common Treatment: QCH
####################################################
pos_actual_h = CacheHeuristic(pilot_pos_games, pilot_pos_row_plays);
# pilot_pos_opp_h = CacheHeuristic(transpose.(pilot_pos_games), pilot_pos_col_plays)
pos_qch_h = QCH(0.07, 0.6, 1.5, 1.7, 1.9)
fit_qch_pos = deepcopy(pos_qch_h)
fit_qch_pos = fit_h!(fit_qch_pos, pilot_pos_games, pos_actual_h)
prediction_loss(fit_qch_pos, pilot_pos_games, pos_actual_h)

m = QCH(9.209684047623952e-8, 0.4043408343646887, 1.2155200432281212, 0.7678744934201345, 1.84556218589128)
prediction_loss(m, pilot_pos_games, pos_actual_h)


fit_qch_pos = deepcopy(pos_qch_h)
fit_qch_pos = fit_h!(fit_qch_pos, pilot_pos_train_games, pos_actual_h)
@show prediction_loss(fit_qch_pos, pilot_pos_train_games, pos_actual_h);
@show prediction_loss(fit_qch_pos, pilot_pos_test_games, pos_actual_h);
@show prediction_loss(fit_qch_pos, pilot_pos_games[comparison_idx], pos_actual_h)

opt_qch_pos = deepcopy(pos_qch_h)
opt_qch_pos = optimize_h!(opt_qch_pos, pilot_pos_train_games, pos_actual_h, costs)
@show prediction_loss(opt_qch_pos, pilot_pos_train_games, pos_actual_h);
@show prediction_loss(opt_qch_pos, pilot_pos_test_games, pos_actual_h);
@show prediction_loss(opt_qch_pos, pilot_pos_games[comparison_idx], pos_actual_h)
####################################################
#%% common Treatment: MetaHeuristic
####################################################
# mh_pos = MetaHeuristic([JointMax(3.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);
mh_pos = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowMean(2.)])], [0., 0., 0.]);
n_fit_iter = 5

fit_mh_pos = deepcopy(mh_pos)
for i in 1:n_fit_iter
    fit_mh_pos = fit_prior!(fit_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs)
    fit_mh_pos = fit_h!(fit_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs; init_x = get_parameters(fit_mh_pos))
end


prediction_loss(fit_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_train_data)
prediction_loss(fit_mh_pos, pilot_pos_test_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_test_data)
prediction_loss(fit_mh_pos, pilot_pos_games[comparison_idx], pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_data[comparison_idx])
h_dists = [h_distribution(fit_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_h_dist = mean(h_dists)


opt_mh_pos = deepcopy(mh_pos)
for i in 1:n_fit_iter
    opt_mh_pos = opt_prior!(opt_mh_pos, pilot_pos_train_games, pos_actual_h, costs)
    opt_mh_pos = optimize_h!(opt_mh_pos, pilot_pos_train_games, pos_actual_h, costs)
end

prediction_loss(opt_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_train_data)
prediction_loss(opt_mh_pos, pilot_pos_test_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_test_data)
prediction_loss(opt_mh_pos, pilot_pos_games[comparison_idx], pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_data[comparison_idx])
h_dists = [h_distribution(opt_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_h_dist = mean(h_dists)





####################################################
#%% common Treatment: Setup and run Deep Heuristic without action response
####################################################
# model_0_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
# model_0_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Last(1))
fit_deep_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))
# model_0_pos = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

# Specific loss function for model_0_pos
# loss(x::Game, y) = Flux.crossentropy(model_0_pos(x), y) + γ*sum(norm, params(model_0_pos))
loss(x::Game, y) = Flux.crossentropy(fit_deep_pos(x), y) + γ*sum(norm, params(fit_deep_pos))
# loss_no_norm(x::Game, y) = Flux.crossentropy(model_0_pos(x), y)
loss_no_norm(x::Game, y) = Flux.crossentropy(fit_deep_pos(x), y)

#%% estimate
# ps = Flux.params(model_0_pos)
ps = Flux.params(fit_deep_pos)
evalcb() = @show(loss_no_norm(pilot_pos_train_data), min_loss(pilot_pos_train_data), loss_no_norm(pilot_pos_test_data), min_loss(pilot_pos_test_data),loss_no_norm(pilot_pos_data[comparison_idx]))
println("clear print are")
5
@epochs n_epochs Flux.train!(loss, ps, pilot_pos_train_data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% common Treatment: Add action response layers
####################################################
# fit_deep_pos = deepcopy(Chain(model_0_pos.layers[1:(length(model_0_pos.layers)-1)]..., Action_Response(1), Last(2)))
#
# loss_no_norm(pilot_pos_data[comparison_idx])
# loss_no_norm(pilot_pos_data)
# loss_no_norm(pilot_pos_test_data)
# #%% Estimate with action response layers
# loss(x::Game, y) = Flux.crossentropy(model_action_pos(x), y) + γ*sum(norm, params(model_action_pos))
# loss_no_norm(x::Game, y) = Flux.crossentropy(model_action_pos(x), y)
# ps = Flux.params(model_action_pos)
#
# 5
# evalcb() = @show(rand_loss(pilot_pos_train_data), loss_no_norm(pilot_pos_train_data), min_loss(pilot_pos_train_data), rand_loss(pilot_pos_test_data), loss_no_norm(pilot_pos_test_data), min_loss(pilot_pos_test_data), loss_no_norm(pilot_pos_data[comparison_idx]))
# println("apa")
# 4
# @epochs 50 Flux.train!(loss, ps, pilot_pos_train_data, opt, cb = Flux.throttle(evalcb,5))



####################################################
#%% common Treatment: Setup and run OPTIMAL Deep Heuristic
####################################################
opt_deep_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))

# Specific loss function for model_0_pos
loss(x::Game, y) = begin
    pred_play = opt_deep_pos(x)
    -expected_payoff(pred_play, pos_actual_h, x) + γ*sum(norm, params(opt_deep_pos)) + exact_cost/Flux.crossentropy(pred_play,pred_play)
    # -expected_payoff(pred_play, pos_actual_h, x) + γ*sum(norm, params(opt_deep_pos)) + sim_cost*my_softmax(opt_deep_pos.layers[end].v)[2] + exact_cost/Flux.crossentropy(pred_play,pred_play)
end
loss_no_norm(x::Game, y) = Flux.crossentropy(opt_deep_pos(x), y)
loss_no_cost(x::Game, y) = -expected_payoff(opt_deep_pos(x), pos_actual_h, x)
loss_no_norm(x::Game, y) = Flux.crossentropy(opt_deep_pos(x), y)
loss_no_cost(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_cost(g,y) for (g,y) in data])/length(data)

#%% estimate
ps = Flux.params(opt_deep_pos)
evalcb() = @show(loss(pilot_pos_train_data), loss_no_cost(pilot_pos_train_data), loss_no_cost(pilot_pos_data), loss_no_norm(pilot_pos_train_data), loss_no_norm(pilot_pos_test_data), loss_no_norm(pilot_pos_data[comparison_idx]))
println("clear print are")
5
@epochs n_epochs Flux.train!(loss, ps, pilot_pos_train_data, opt, cb = Flux.throttle(evalcb,5))

###########################################################################
#%% competing Treatment: Load the data
###########################################################################
comparison_idx = [31, 37, 41, 44, 49]
later_com = 50 .+ comparison_idx
comparison_idx = [comparison_idx..., later_com...]

pilot_neg_data = [(json_to_game(first(competing_games_df[competing_games_df.round .== i, :row])), convert(Vector{Float64}, JSON.parse(first(competing_games_df[competing_games_df.round .== i, :row_play])))) for i in 1:50];
append!(pilot_neg_data ,[(json_to_game(first(competing_games_df[competing_games_df.round .== i, :col])), convert(Vector{Float64}, JSON.parse(first(competing_games_df[competing_games_df.round .== i, :col_play])))) for i in 1:50]);

# By adding a minor random noise to the only symmetric game we can distinguish the game from the row and columns perspective
pilot_neg_data[41] = (Game(pilot_neg_data[41][1].row .+ rand(3,3)*0.01, pilot_neg_data[41][1].col .+ rand(3,3)*0.01), pilot_neg_data[41][2])
pilot_neg_data[91] = (transpose(pilot_neg_data[41][1]), pilot_neg_data[91][2])


pilot_neg_games = [d[1] for d in pilot_neg_data];
pilot_neg_row_plays = [d[2] for d in pilot_neg_data];
pilot_neg_col_plays = [pilot_neg_row_plays[51:100]..., pilot_neg_row_plays[1:50]...]

pilot_neg_n_train = 70
pilot_neg_train_idxs = sample(1:length(pilot_neg_games), pilot_neg_n_train; replace=false)
pilot_neg_test_idxs = setdiff(1:length(pilot_neg_games), pilot_neg_train_idxs)
# pilot_neg_train_idxs = setdiff(1:length(pilot_neg_games), comparison_idx)
# pilot_neg_test_idxs = comparison_idx
sort!(pilot_neg_train_idxs)
sort!(pilot_neg_test_idxs)
pilot_neg_train_games = pilot_neg_games[pilot_neg_train_idxs];
pilot_neg_test_games = pilot_neg_games[pilot_neg_test_idxs];
pilot_neg_train_row = pilot_neg_row_plays[pilot_neg_train_idxs]
pilot_neg_test_row = pilot_neg_row_plays[pilot_neg_test_idxs]
pilot_neg_train_data = pilot_neg_data[pilot_neg_train_idxs];
pilot_neg_test_data = pilot_neg_data[pilot_neg_test_idxs];

####################################################
#%% competing Treatment: QCH
####################################################
neg_actual_h = CacheHeuristic(pilot_neg_games, pilot_neg_row_plays);
neg_qch_h = QCH(0.07, 0.64, 1.5, 1.7, 1.9)


fit_qch_neg = deepcopy(neg_qch_h)
fit_qch_neg = fit_h!(fit_qch_neg, pilot_neg_train_games, neg_actual_h)
prediction_loss(fit_qch_neg, pilot_neg_train_games, neg_actual_h)
prediction_loss(fit_qch_neg, pilot_neg_test_games, neg_actual_h)
prediction_loss(fit_qch_neg, pilot_neg_games[comparison_idx], neg_actual_h)

opt_qch_neg = deepcopy(neg_qch_h)
opt_qch_neg = optimize_h!(opt_qch_neg, pilot_neg_train_games, neg_actual_h, costs)
prediction_loss(opt_qch_neg, pilot_neg_train_games, neg_actual_h)
prediction_loss(opt_qch_neg, pilot_neg_test_games, neg_actual_h)
prediction_loss(opt_qch_neg, pilot_neg_games[comparison_idx], neg_actual_h)


####################################################
#%% competing Treatment: MetaHeuristic
####################################################
# mh_neg = MetaHeuristic([JointMax(3.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);
mh_neg = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(0., 1.), RowMean( 2.)])], [0., 0., 0.]);


fit_mh_neg = deepcopy(mh_neg)
for i in 1:n_fit_iter
    fit_mh_neg = fit_prior!(fit_mh_neg, pilot_neg_train_games, neg_actual_h, neg_actual_h, costs)
    fit_mh_neg = fit_h!(fit_mh_neg, pilot_neg_train_games, neg_actual_h, neg_actual_h, costs; init_x = get_parameters(fit_mh_neg))
end

prediction_loss(fit_mh_neg, pilot_neg_games, neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_data)
prediction_loss(fit_mh_neg, pilot_neg_train_games, neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_train_data)
prediction_loss(fit_mh_neg, pilot_neg_test_games, neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_test_data)
prediction_loss(fit_mh_neg, pilot_neg_games[comparison_idx], neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_data[comparison_idx])
h_dists = [h_distribution(fit_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games];
avg_h_dist = mean(h_dists)

opt_mh_neg = deepcopy(mh_neg)
for i in 1:n_fit_iter
    opt_mh_neg = opt_prior!(opt_mh_neg, pilot_neg_train_games, neg_actual_h, costs)
    opt_mh_neg = optimize_h!(opt_mh_neg, pilot_neg_train_games, neg_actual_h, costs)
end

prediction_loss(opt_mh_neg, pilot_neg_games, neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_data)
prediction_loss(opt_mh_neg, pilot_neg_train_games, neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_train_data)
prediction_loss(opt_mh_neg, pilot_neg_test_games, neg_actual_h, neg_actual_h, costs)
min_loss(pilot_neg_test_data)
prediction_loss(opt_mh_neg, pilot_neg_games[comparison_idx], neg_actual_h, neg_actual_h, costs)

min_loss(pilot_neg_data[comparison_idx])
h_dists = [h_distribution(opt_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games];
avg_h_dist = mean(h_dists)
#
# mh_neg = MetaHeuristic([JointMax(1.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);
#
# costs = Costs(0.7, 0.1, 0.2, 2.5)
# opt_mh_neg = mh_neg
# opt_mh_neg = fit_prior!(opt_mh_neg, pilot_neg_train_games, neg_actual_h, neg_actual_h, costs)
# opt_mh_neg = fit_h!(opt_mh_neg, pilot_neg_train_games, neg_actual_h, neg_actual_h, costs; init_x = get_parameters(mh))
#
#
# prediction_loss(opt_mh_neg, pilot_neg_games, neg_actual_h)
# min_loss(pilot_neg_data)
#
# prediction_loss(opt_mh_neg, pilot_neg_train_games, neg_actual_h)
# prediction_loss(opt_mh_neg, pilot_neg_test_games, neg_actual_h)
# min_loss(pilot_neg_test_data)
# prediction_loss(opt_mh_neg, pilot_neg_games[comparison_idx], neg_actual_h)
#
# h_dists = [h_distribution(opt_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games]
# avg_h_dist = mean(h_dists)

#%% Compare common and competing
println("-------- Meta Heuristics fitted to data -----------")
println("neg on neg", prediction_loss(fit_mh_neg, pilot_neg_games, neg_actual_h))
println("pos on neg", prediction_loss(fit_mh_pos, pilot_neg_games, neg_actual_h))

println("neg on pos", prediction_loss(fit_mh_neg, pilot_pos_games, pos_actual_h))
println("pos on pos", prediction_loss(fit_mh_pos, pilot_pos_games, pos_actual_h))

println("-------- Optimal Meta Heuristics  -----------")
println("neg on neg", prediction_loss(opt_mh_neg, pilot_neg_games, neg_actual_h))
println("pos on neg", prediction_loss(opt_mh_pos, pilot_neg_games, neg_actual_h))

println("neg on pos", prediction_loss(opt_mh_neg, pilot_pos_games, pos_actual_h))
println("pos on pos", prediction_loss(opt_mh_pos, pilot_pos_games, pos_actual_h))


####################################################
#%% competing Treatment: Setup and run Deep Heuristic without action response
####################################################
# model_0_neg = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
fit_deep_neg = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Last(1))
# fit_deep_neg = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

# Specific loss function for fit_deep_neg
loss(x::Game, y) = Flux.crossentropy(fit_deep_neg(x), y) + γ*sum(norm, params(fit_deep_neg))
loss_no_norm(x::Game, y) = Flux.crossentropy(fit_deep_neg(x), y)

#%% estimate
ps = Flux.params(fit_deep_neg)
evalcb() = @show(rand_loss(pilot_neg_train_data), loss_no_norm(pilot_neg_train_data), min_loss(pilot_neg_train_data), rand_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_test_data), min_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_data[comparison_idx]), loss_no_norm(pilot_pos_data))
println("clear print are")
5
@epochs n_epochs Flux.train!(loss, ps, pilot_neg_train_data, opt, cb = Flux.throttle(evalcb,5))





####################################################
#%% competing Treatment: Add action response layers
####################################################
# fit_deep_neg = deepcopy(Chain(model_0_neg.layers[1:(length(model_0_neg.layers)-1)]..., Action_Response(1), Last(2)))
# loss(x::Game, y) = Flux.crossentropy(model_action_neg(x), y) + γ*sum(norm, params(model_action_neg))
# loss_no_norm(x::Game, y) = Flux.crossentropy(model_action_neg(x), y)
#
# loss_no_norm(pilot_neg_data[comparison_idx])
# loss_no_norm(pilot_neg_data)
# #%% Estimate with action response layers
# ps = Flux.params(model_action_neg)
# opt = ADAM(0.0001, (0.9, 0.999))
# evalcb() = @show(rand_loss(pilot_neg_train_data), loss_no_norm(pilot_neg_train_data), min_loss(pilot_neg_train_data), rand_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_test_data), min_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_data[comparison_idx]),loss_no_norm(pilot_pos_data))
# println("apa")
# 4
# @epochs 50 Flux.train!(loss, ps, pilot_neg_train_data, opt, cb = Flux.throttle(evalcb,5))

####################################################
#%% competing Treatment: Setup and run OPTIMAL Deep Heuristic without action response
####################################################
# model_0_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
opt_deep_neg = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Action_Response(1), Last(2))
# model_0_neg = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

# Specific loss function for model_0_neg
loss(x::Game, y) = begin
    pred_play = opt_deep_neg(x)
    -expected_payoff(pred_play, neg_actual_h, x) + γ*sum(norm, params(opt_deep_neg)) + exact_cost/Flux.crossentropy(pred_play,pred_play)
    # -expected_payoff(pred_play, neg_actual_h, x) + γ*sum(norm, params(opt_deep_neg)) + sim_cost*my_softmax(opt_deep_neg.layers[end].v)[2] + exact_cost/Flux.crossentropy(pred_play,pred_play)
end
loss_no_norm(x::Game, y) = Flux.crossentropy(opt_deep_neg(x), y)
loss_no_cost(x::Game, y) = -expected_payoff(opt_deep_neg(x), neg_actual_h, x)
loss_no_norm(x::Game, y) = Flux.crossentropy(opt_deep_neg(x), y)
loss_no_cost(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_cost(g,y) for (g,y) in data])/length(data)

#%% estimate
ps = Flux.params(opt_deep_neg)
evalcb() = @show(loss(pilot_neg_data), loss_no_cost(pilot_neg_data), rand_loss(pilot_neg_data), loss_no_norm(pilot_neg_data), min_loss(pilot_neg_data), loss_no_norm(pilot_neg_data[comparison_idx]), loss_no_norm(pilot_pos_data))
println("clear print are")
5
@epochs n_epochs Flux.train!(loss, ps, pilot_neg_train_data, opt, cb = Flux.throttle(evalcb,5))


######################################################
#%% Compare the estimated heuristics
######################################################

println("------- QCH ----------")
println("fit Neg QCH: ", fit_qch_neg)
println("fit Pos QCH: ", fit_qch_pos)
println("fit neg on neg: ", prediction_loss(fit_qch_neg, pilot_neg_games, neg_actual_h))
println("fit pos on neg: ", prediction_loss(fit_qch_pos, pilot_neg_games, neg_actual_h))
println("fit neg on neg comp: ", prediction_loss(fit_qch_neg, pilot_neg_games[comparison_idx], neg_actual_h))
println("fit pos on neg comp: ", prediction_loss(fit_qch_pos, pilot_neg_games[comparison_idx], neg_actual_h))
println("fit neg on pos: ", prediction_loss(fit_qch_neg, pilot_pos_games, pos_actual_h))
println("fit pos on pos: ", prediction_loss(fit_qch_pos, pilot_pos_games, pos_actual_h))
println("fit neg on pos comp: ", prediction_loss(fit_qch_neg, pilot_pos_games[comparison_idx], pos_actual_h))
println("fit pos on pos comp: ", prediction_loss(fit_qch_pos, pilot_pos_games[comparison_idx], pos_actual_h))

println("opt Neg QCH: ", opt_qch_neg)
println("opt Pos QCH: ", opt_qch_pos)
println("opt neg on neg: ", prediction_loss(opt_qch_neg, pilot_neg_games, neg_actual_h))
println("opt pos on neg: ", prediction_loss(opt_qch_pos, pilot_neg_games, neg_actual_h))
println("opt neg on neg comp: ", prediction_loss(opt_qch_neg, pilot_neg_games[comparison_idx], neg_actual_h))
println("opt pos on neg comp: ", prediction_loss(opt_qch_pos, pilot_neg_games[comparison_idx], neg_actual_h))
println("opt neg on pos: ", prediction_loss(opt_qch_neg, pilot_pos_games, pos_actual_h))
println("opt pos on pos: ", prediction_loss(opt_qch_pos, pilot_pos_games, pos_actual_h))
println("opt neg on pos comp: ", prediction_loss(opt_qch_neg, pilot_pos_games[comparison_idx], pos_actual_h))
println("opt pos on pos comp: ", prediction_loss(opt_qch_pos, pilot_pos_games[comparison_idx], pos_actual_h))

println("------- Meta Heuristics ----------")
println("Neg MH: ", fit_mh_neg)
fit_h_dists_neg = [h_distribution(fit_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games];
avg_fit_h_dist_neg = mean(fit_h_dists_neg);
println("Avg h neg: ", avg_fit_h_dist_neg)
println("Pos MH: ", fit_mh_pos)
fit_h_dists_pos = [h_distribution(fit_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_fit_h_dist_pos = mean(fit_h_dists_pos);
println("Avg h pos: ", avg_fit_h_dist_pos)
println("neg on neg: ", prediction_loss(fit_mh_neg, pilot_neg_games, neg_actual_h))
println("pos on neg: ", prediction_loss(fit_mh_pos, pilot_neg_games, neg_actual_h))
println("neg on neg comp: ", prediction_loss(fit_mh_neg, pilot_neg_games[comparison_idx], neg_actual_h))
println("pos on neg comp: ", prediction_loss(fit_mh_pos, pilot_neg_games[comparison_idx], neg_actual_h))

println("neg on pos: ", prediction_loss(fit_mh_neg, pilot_pos_games, pos_actual_h))
println("pos on pos: ", prediction_loss(fit_mh_pos, pilot_pos_games, pos_actual_h))
println("neg on pos comp: ", prediction_loss(fit_mh_neg, pilot_pos_games[comparison_idx], pos_actual_h))
println("pos on pos comp: ", prediction_loss(fit_mh_pos, pilot_pos_games[comparison_idx], pos_actual_h))


println("------- Optimal Meta Heuristics ----------")
println("Neg MH: ", opt_mh_neg)
opt_h_dists_neg = [h_distribution(opt_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games];
avg_opt_h_dist_neg = mean(opt_h_dists_neg);
println("Avg h neg: ", avg_opt_h_dist_neg)
println("Pos MH: ", opt_mh_pos)
opt_h_dists_pos = [h_distribution(opt_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_opt_h_dist_pos = mean(opt_h_dists_pos);
println("Avg h pos: ", avg_opt_h_dist_pos)
println("neg on neg: ", prediction_loss(opt_mh_neg, pilot_neg_games, neg_actual_h))
println("pos on neg: ", prediction_loss(opt_mh_pos, pilot_neg_games, neg_actual_h))
println("neg on neg comp: ", prediction_loss(opt_mh_neg, pilot_neg_games[comparison_idx], neg_actual_h))
println("pos on neg comp: ", prediction_loss(opt_mh_pos, pilot_neg_games[comparison_idx], neg_actual_h))

println("neg on pos: ", prediction_loss(opt_mh_neg, pilot_pos_games, pos_actual_h))
println("pos on pos: ", prediction_loss(opt_mh_pos, pilot_pos_games, pos_actual_h))
println("neg on pos comp: ", prediction_loss(opt_mh_neg, pilot_pos_games[comparison_idx], pos_actual_h))
println("pos on pos comp: ", prediction_loss(opt_mh_pos, pilot_pos_games[comparison_idx], pos_actual_h))

println("------- Deep Heuristics ----------")
# loss_no_norm_neg(x::Game, y) = Flux.crossentropy(model_0_neg(x), y);
loss_no_norm_neg(x::Game, y) = Flux.crossentropy(fit_deep_neg(x), y);
loss_no_norm_neg(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm_neg(g,y) for (g,y) in data])/length(data);

loss_no_norm_pos(x::Game, y) = Flux.crossentropy(fit_deep_pos(x), y);
# loss_no_norm_pos(x::Game, y) = Flux.crossentropy(model_0_pos(x), y);
loss_no_norm_pos(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm_pos(g,y) for (g,y) in data])/length(data);
println("neg on neg: ", loss_no_norm_neg(pilot_neg_data))
println("pos on neg: ", loss_no_norm_pos(pilot_neg_data))
println("neg on neg comp: ", loss_no_norm_neg(pilot_neg_data[comparison_idx]))
println("pos on neg comp: ", loss_no_norm_pos(pilot_neg_data[comparison_idx]))

println("neg on pos: ", loss_no_norm_neg(pilot_pos_data))
println("pos on pos: ", loss_no_norm_pos(pilot_pos_data))
println("neg on pos comp: ", loss_no_norm_neg(pilot_pos_data[comparison_idx]))
println("pos on pos comp: ", loss_no_norm_pos(pilot_pos_data[comparison_idx]))


println("------- Optimal Deep Heuristics ----------")
loss_no_norm_neg_opt(x::Game, y) = Flux.crossentropy(opt_deep_neg(x), y);
loss_no_norm_neg_opt(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm_neg_opt(g,y) for (g,y) in data])/length(data);

loss_no_norm_pos_opt(x::Game, y) = Flux.crossentropy(opt_deep_pos(x), y);
loss_no_norm_pos_opt(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm_pos_opt(g,y) for (g,y) in data])/length(data);
println("neg on neg: ", loss_no_norm_neg_opt(pilot_neg_data))
println("pos on neg: ", loss_no_norm_pos_opt(pilot_neg_data))
println("neg on neg comp: ", loss_no_norm_neg_opt(pilot_neg_data[comparison_idx]))
println("pos on neg comp: ", loss_no_norm_pos_opt(pilot_neg_data[comparison_idx]))

println("neg on pos: ", loss_no_norm_neg_opt(pilot_pos_data))
println("pos on pos: ", loss_no_norm_pos_opt(pilot_pos_data))
println("neg on pos comp: ", loss_no_norm_neg_opt(pilot_pos_data[comparison_idx]))
println("pos on pos comp: ", loss_no_norm_pos_opt(pilot_pos_data[comparison_idx]))



######################################################
#%% Print meta heuristics to latex_snippets
######################################################

opt_mh_pos.h_list[1].λ

estimated_mh_table = """
\\begin{tabular}{@{}l l c  c  c }
% & & \\multicolumn{3}{c}{Estimated} \\\\
  \\multicolumn{5}{c}{\\textbf{Estimated Meta Heuristic}}   \\\\ \\cline{1-5}
& & \\multicolumn{1}{ c}{Jointmax}  & Row Heuristic & \\multicolumn{1}{c}{Sim}  \\\\
& & \\multicolumn{1}{ c}{\$\\varphi\$}  & \$\\gamma, \\varphi\$ & \\multicolumn{1}{c}{\$(\\gamma, \\varphi), \\varphi\$}  \\\\ \\cline{1-5}
\\multicolumn{1}{l|}{\\multirow{2}{*}{Common interests}} & \\multicolumn{1}{l|}{Params} & $(round(fit_mh_pos.h_list[1].λ, digits=2)) & $(round(fit_mh_pos.h_list[2].γ, digits=2)), $(round(fit_mh_pos.h_list[2].λ, digits=2)) & ($(round(fit_mh_pos.h_list[3].h_list[1].γ, digits=2)), $(round(fit_mh_pos.h_list[3].h_list[1].λ, digits=2))), $(round(fit_mh_pos.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_fit_h_dist_pos[1]*100)) \\% & $(round(Int, avg_fit_h_dist_pos[2]*100)) \\% & $(round(Int, avg_fit_h_dist_pos[3]*100)) \\% \\\\
\\cline{1-5}\\multicolumn{1}{l|}{\\multirow{2}{*}{Competing interests}} &\\multicolumn{1}{l|}{Params} & $(round(fit_mh_neg.h_list[1].λ, digits=2)) & $(round(fit_mh_neg.h_list[2].γ, digits=2)), $(round(fit_mh_neg.h_list[2].λ, digits=2)) & ($(round(fit_mh_neg.h_list[3].h_list[1].γ, digits=2)), $(round(fit_mh_neg.h_list[3].h_list[1].λ, digits=2))), $(round(fit_mh_neg.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_fit_h_dist_neg[1]*100)) \\% & $(round(Int, avg_fit_h_dist_neg[2]*100)) \\% & $(round(Int, avg_fit_h_dist_neg[3]*100)) \\% \\\\ \\cline{1-5}
\\end{tabular}
"""
println(estimated_mh_table)

open("../overleaf/tex_snippets/estim_meta_heuristics.tex","w") do f
    write(f, estimated_mh_table)
end

optimal_mh_table = """
\\begin{tabular}{@{}l l c  c  c }
% & & \\multicolumn{3}{c}{Estimated} \\\\
  \\multicolumn{5}{c}{\\textbf{Optimal Meta Heuristic}}   \\\\ \\cline{1-5}
& & \\multicolumn{1}{ c}{Jointmax}  & Row Heuristic & \\multicolumn{1}{c}{Sim}  \\\\
& & \\multicolumn{1}{ c}{\$\\varphi\$}  & \$\\gamma, \\varphi\$ & \\multicolumn{1}{c}{\$(\\gamma, \\varphi), \\varphi\$}  \\\\ \\cline{1-5}
\\multicolumn{1}{l|}{\\multirow{2}{*}{Common interests}} & \\multicolumn{1}{l|}{Params} & $(round(opt_mh_pos.h_list[1].λ, digits=2)) & $(round(opt_mh_pos.h_list[2].γ, digits=2)), $(round(opt_mh_pos.h_list[2].λ, digits=2)) & ($(round(opt_mh_pos.h_list[3].h_list[1].γ, digits=2)), $(round(opt_mh_pos.h_list[3].h_list[1].λ, digits=2))), $(round(opt_mh_pos.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_opt_h_dist_pos[1]*100)) \\% & $(round(Int, avg_opt_h_dist_pos[2]*100)) \\% & $(round(Int, avg_opt_h_dist_pos[3]*100)) \\% \\\\
\\cline{1-5}\\multicolumn{1}{l|}{\\multirow{2}{*}{Competing interests}} &\\multicolumn{1}{l|}{Params} & $(round(opt_mh_neg.h_list[1].λ, digits=2)) & $(round(opt_mh_neg.h_list[2].γ, digits=2)), $(round(opt_mh_neg.h_list[2].λ, digits=2)) & ($(round(opt_mh_neg.h_list[3].h_list[1].γ, digits=2)), $(round(opt_mh_neg.h_list[3].h_list[1].λ, digits=2))), $(round(opt_mh_neg.h_list[3].h_list[2].λ, digits=2)) \\\\ \\cline{2-5}
\\multicolumn{1}{l|}{}                        & \\multicolumn{1}{l|}{Share}  & $(round(Int, avg_opt_h_dist_neg[1]*100)) \\% & $(round(Int, avg_opt_h_dist_neg[2]*100)) \\% & $(round(Int, avg_opt_h_dist_neg[3]*100)) \\% \\\\ \\cline{1-5}
\\end{tabular}
"""
println(optimal_mh_table)


open("../overleaf/tex_snippets/opt_meta_heuristics.tex","w") do f
    write(f, optimal_mh_table)
end

######################################################
#%% Generate Data Frame
######################################################
data_dict = Dict()
data_dict["Competing"] = Dict(
"all" => Dict("data" => pilot_neg_data, "games" => pilot_neg_games, "row_play" => pilot_neg_row_plays),
"train" => Dict("data" => pilot_neg_train_data, "games" => pilot_neg_train_games, "row_play" => pilot_neg_train_row), "test" => Dict("data" => pilot_neg_test_data, "games" => pilot_neg_test_games, "row_play" => pilot_neg_test_row),
"comparison" => Dict("data" => pilot_neg_data[comparison_idx], "games" => pilot_neg_games[comparison_idx], "row_play" => pilot_neg_row_plays[comparison_idx]),
"actual_h" => neg_actual_h);

data_dict["Common"] = Dict(
"all" => Dict("data" => pilot_pos_data, "games" => pilot_pos_games, "row_play" => pilot_pos_row_plays),
"train" => Dict("data" => pilot_pos_train_data, "games" => pilot_pos_train_games, "row_play" => pilot_pos_train_row), "test" => Dict("data" => pilot_pos_test_data, "games" => pilot_pos_test_games, "row_play" => pilot_pos_test_row),
"comparison" => Dict("data" => pilot_pos_data[comparison_idx], "games" => pilot_pos_games[comparison_idx], "row_play" => pilot_pos_row_plays[comparison_idx]),
"actual_h" => pos_actual_h);


res_df = DataFrame()

for treatment in ["Competing", "Common"]
    for data_type in ["all", "train", "test", "comparison"]
        res_dict = Dict{Any, Any}(:data_type => data_type, :treatment => treatment)
        data = data_dict[treatment][data_type]
        actual = data_dict[treatment]["actual_h"]
        res_dict[:random] = rand_loss(data["data"])
        res_dict[:minimum] = min_loss(data["data"])
        res_dict[:fit_QCH_pos] = prediction_loss(fit_qch_pos, data["games"], actual)
        res_dict[:fit_QCH_neg] = prediction_loss(fit_qch_neg, data["games"], actual)
        res_dict[:opt_QCH_pos] = prediction_loss(opt_qch_pos, data["games"], actual)
        res_dict[:opt_QCH_neg] = prediction_loss(opt_qch_neg, data["games"], actual)
        res_dict[:fit_mh_neg] = prediction_loss(fit_mh_neg, data["games"], actual)
        res_dict[:fit_mh_pos] = prediction_loss(fit_mh_pos, data["games"], actual)
        res_dict[:opt_mh_neg] = prediction_loss(opt_mh_neg, data["games"], actual)
        res_dict[:opt_mh_pos] = prediction_loss(opt_mh_pos, data["games"], actual)
        res_dict[:opt_deep_neg] = loss_no_norm_neg_opt(data["data"]).data
        res_dict[:opt_deep_pos] = loss_no_norm_pos_opt(data["data"]).data
        res_dict[:fit_deep_neg] = loss_no_norm_neg(data["data"]).data
        res_dict[:fit_deep_pos] = loss_no_norm_pos(data["data"]).data
        if length(names(res_df)) == 0
            res_df = DataFrame(res_dict)
        else
            push!(res_df, res_dict)
        end
    end
end

res_df

CSV.write("res_df_from_pilot.csv",res_df)
###########################################################
#%% Plot the relative likelihoods
###########################################################

using Plots
using StatsPlots
pyplot()


data_names = [:random, :fit_QCH_neg, :fit_QCH_pos, :opt_QCH_neg, :opt_QCH_pos, :fit_mh_neg, :fit_mh_pos, :opt_mh_neg, :opt_mh_pos, :fit_deep_neg,  :fit_deep_pos, :opt_deep_neg, :opt_deep_pos, :minimum]

plots_vec = []
for data_type in ["train", "test", "comparison"], treat in ["Common", "Competing"]
    vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), data_names]))
    ctg = [repeat(["competing"], 6)..., repeat(["common"], 6)...]
    nam = [repeat(["fit QCH", "opt QCH", "fit mh", "opt mh", "fit deep", "opt deep"],2)...]
    bar_vals = hcat(transpose(reshape(vals[2:(end-1)], (2,6))))
    plt = groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*" interest "*data_type*" games", ylims=(0,1.3))
    plot!([vals[1]], linetype=:hline, width=2, label="random loss", color=:grey)
    plot!([vals[14]], linetype=:hline, width=2, label="min loss", color=:black)
    push!(plots_vec, plt)
end

length(plots_vec)
plot(plots_vec..., layout=(3,2), size=(794,1123))

savefig("../overleaf/figs/test.png")

res_df



###########################################################
#%% Compare payoffs over time
###########################################################
function payoff(p, p_opp, g::Game)
    p_outcome = p * p_opp'
    sum(p_outcome .* g.row)
end

function max_payoff(p_opp, g::Game)
    maximum(sum(g.row .* p_opp', dims=2))
end

function rand_payoff(p_opp, g::Game)
    mean(sum(g.row .* p_opp', dims=2))
end

payoff_pos_df = DataFrame()

joint_h = JointMax(10.)
hrow_pos = opt_mh_pos.h_list[2]
hrow_pos = optimize_h!(hrow_pos, pilot_pos_games, pos_actual_h, costs)
hrow_neg = opt_mh_neg.h_list[2]
hrow_neg = optimize_h!(hrow_neg, pilot_neg_games, neg_actual_h, costs)
h_max = RowHeuristic(10., 10.)
h_mean = RowHeuristic(0., 10.)
h_min = RowHeuristic(-10., 10.)
sim_h = SimHeuristic([RowHeuristic(0., 10.), RowHeuristic(0., 10.)])

function gen_payoff_data(row, col, game)
    res_dict = Dict()

    res_dict[:row_max] = max_payoff(col, game)
    res_dict[:row_rand] = rand_payoff(col, game)
    res_dict[:row_payoff] = payoff(row, col, game)
    res_dict[:row_opt_deep_pos] = payoff(opt_deep_pos(game), col, game).data
    res_dict[:row_opt_deep_neg] = payoff(opt_deep_neg(game), col, game).data
    res_dict[:row_fit_deep_pos] = payoff(fit_deep_pos(game), col, game).data
    res_dict[:row_fit_deep_neg] = payoff(fit_deep_neg(game), col, game).data
    res_dict[:row_opt_mh_pos] = payoff(play_distribution(opt_mh_pos, game), col, game)
    res_dict[:row_opt_mh_neg] = payoff(play_distribution(opt_mh_neg, game), col, game)
    res_dict[:row_fit_mh_pos] = payoff(play_distribution(fit_mh_pos, game), col, game)
    res_dict[:row_fit_mh_neg] = payoff(play_distribution(fit_mh_neg, game), col, game)
    res_dict[:row_joint_h] = payoff(play_distribution(joint_h, game), col, game)
    res_dict[:row_h_max] = payoff(play_distribution(h_max, game), col, game)
    res_dict[:row_h_min] = payoff(play_distribution(h_min, game), col, game)
    res_dict[:row_h_mean] = payoff(play_distribution(h_mean, game), col, game)
    res_dict[:row_sim_h] = payoff(play_distribution(sim_h, game), col, game)
    res_dict[:row_hrow_pos] = payoff(play_distribution(hrow_pos, game), col, game)
    res_dict[:row_hrow_neg] = payoff(play_distribution(hrow_neg, game), col, game)


    col_game = transpose(game)
    res_dict[:col_max] = max_payoff(row, col_game)
    res_dict[:col_rand] = rand_payoff(row, col_game)
    res_dict[:col_payoff] = payoff(col, row, col_game)
    res_dict[:col_opt_deep_pos] = payoff(opt_deep_pos(col_game), row, col_game).data
    res_dict[:col_opt_deep_neg] = payoff(opt_deep_neg(col_game), row, col_game).data
    res_dict[:col_fit_deep_pos] = payoff(fit_deep_pos(col_game), row, col_game).data
    res_dict[:col_fit_deep_neg] = payoff(fit_deep_neg(col_game), row, col_game).data
    res_dict[:col_opt_mh_pos] = payoff(play_distribution(opt_mh_pos, col_game), row, col_game)
    res_dict[:col_opt_mh_neg] = payoff(play_distribution(opt_mh_neg, col_game), row, col_game)
    res_dict[:col_fit_mh_pos] = payoff(play_distribution(fit_mh_pos, col_game), row, col_game)
    res_dict[:col_fit_mh_neg] = payoff(play_distribution(fit_mh_neg, col_game), row, col_game)
    res_dict[:col_joint_h] = payoff(play_distribution(joint_h, col_game), row, col_game)
    res_dict[:col_h_max] = payoff(play_distribution(h_max, col_game), row, col_game)
    res_dict[:col_h_min] = payoff(play_distribution(h_min, col_game), row, col_game)
    res_dict[:col_h_mean] = payoff(play_distribution(h_mean, col_game), row, col_game)
    res_dict[:col_sim_h] = payoff(play_distribution(sim_h, col_game), row, col_game)
    res_dict[:col_hrow_pos] = payoff(play_distribution(hrow_pos, col_game), row, col_game)
    res_dict[:col_hrow_neg] = payoff(play_distribution(hrow_neg, col_game), row, col_game)

    res_dict[:avg_max] = (res_dict[:col_max] + res_dict[:row_max])/2
    res_dict[:avg_rand] = (res_dict[:col_rand] + res_dict[:row_rand])/2
    res_dict[:avg_payoff] = (res_dict[:col_payoff] + res_dict[:row_payoff])/2
    res_dict[:avg_opt_deep_pos] = (res_dict[:col_opt_deep_pos] + res_dict[:row_opt_deep_pos])/2
    res_dict[:avg_opt_deep_neg] = (res_dict[:col_opt_deep_neg] + res_dict[:row_opt_deep_neg])/2
    res_dict[:avg_fit_deep_pos] = (res_dict[:col_fit_deep_pos] + res_dict[:row_fit_deep_pos])/2
    res_dict[:avg_fit_deep_neg] = (res_dict[:col_fit_deep_neg] + res_dict[:row_fit_deep_neg])/2
    res_dict[:avg_opt_mh_pos] = (res_dict[:col_opt_mh_pos] + res_dict[:row_opt_mh_pos])/2
    res_dict[:avg_opt_mh_neg] = (res_dict[:col_opt_mh_neg] + res_dict[:row_opt_mh_neg])/2
    res_dict[:avg_fit_mh_pos] = (res_dict[:col_fit_mh_pos] + res_dict[:row_fit_mh_pos])/2
    res_dict[:avg_fit_mh_neg] = (res_dict[:col_fit_mh_neg] + res_dict[:row_fit_mh_neg])/2
    res_dict[:avg_joint_h] = (res_dict[:row_joint_h] + res_dict[:col_joint_h])/2
    res_dict[:avg_h_max] = (res_dict[:row_h_max] + res_dict[:col_h_max])/2
    res_dict[:avg_h_min] = (res_dict[:row_h_min] + res_dict[:col_h_min])/2
    res_dict[:avg_h_mean] = (res_dict[:row_h_mean] + res_dict[:col_h_mean])/2
    res_dict[:avg_sim_h] = (res_dict[:row_sim_h] + res_dict[:col_sim_h])/2
    res_dict[:avg_hrow_pos] = (res_dict[:row_hrow_pos] + res_dict[:col_hrow_pos])/2
    res_dict[:avg_hrow_neg] = (res_dict[:row_hrow_neg] + res_dict[:col_hrow_neg])/2
    return res_dict
end




payoff_pos_df = DataFrame()
for i in 1:50
    row = pilot_pos_row_plays[i]
    col = pilot_pos_col_plays[i]
    game = pilot_pos_games[i]
    to_add = gen_payoff_data(row, col, game)
    to_add[:round] = i
    if i == 1
        payoff_pos_df = DataFrame(to_add)
    else
        append!(payoff_pos_df, to_add)
    end
end

payoff_neg_df = DataFrame()
for i in 1:50
    row = pilot_neg_row_plays[i]
    col = pilot_neg_col_plays[i]
    game = pilot_neg_games[i]
    to_add = gen_payoff_data(row, col, game)
    to_add[:round] = i
    if i == 1
        payoff_neg_df = DataFrame(to_add)
    else
        append!(payoff_neg_df, to_add)
    end
end




using RollingFunctions

@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_payoff, 1), legend=false))
savefig("../overleaf/figs/Commmon_max-actual.png")
@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_payoff, 10), legend=false))
savefig("../overleaf/figs/Commmon_max-actual-10-period.png")
@with(payoff_neg_df, plot(:round, runmean(:avg_max .- :avg_payoff, 1), legend=false))
savefig("../overleaf/figs/Competing_max-actual.png")
@with(payoff_neg_df, plot(:round, runmean(:avg_max .- :avg_payoff, 10), legend=false))
savefig("../overleaf/figs/Competing_max-actual-10-period.png")



@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_opt_deep_pos, 10)))
@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_rand, 10)))
@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_mh_pos, 10)))
@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_mh_neg, 10)))
@with(payoff_pos_df, plot(:round, runmean(:avg_max .- :avg_mh_neg, 10)))

long_pos_payoff_df = melt(payoff_pos_df, [:round])

mean_pos_df = by(long_pos_payoff_df, :variable, common_payoff = :value => mean)

occursin.("avg", string.(mean_pos_df.variable))
avg_res_pos = @where(mean_pos_df, occursin.("avg", string.(:variable)))

long_neg_payoff_df = melt(payoff_neg_df, [:round])

mean_neg_df = by(long_neg_payoff_df, :variable, competitive_payoff = :value => mean)

occursin.("avg", string.(mean_neg_df.variable))
avg_res_neg = @where(mean_neg_df, occursin.("avg", string.(:variable)))

perf_both_df = join(avg_res_pos, avg_res_neg, on=:variable)


perf_both_df.variable = string.(perf_both_df.variable)

@byrow! perf_both_df begin
    :variable = replace(:variable, "avg_" => "")
    :variable = replace(:variable, "_" => " ")
end


using LaTeXTabulars
using LaTeXStrings

name_list = filter(x -> !(x in ["max", "payoff"]), perf_both_df.variable)
@where(perf_both_df, :variable .== name_list[1]).common_payoff[1]


function gen_row(namn)
    comm = round(@where(perf_both_df, :variable .== namn).common_payoff[1], digits=2)
    comp = round(@where(perf_both_df, :variable .== namn).competitive_payoff[1], digits=2)
    [namn comm comp]
end

res_mat = reshape(gen_row(name_list[1]), (1,3))
for namn in name_list[2:end]
    res_mat = vcat(res_mat, gen_row(namn))
end
println(res_mat)

latex_tabular("./../overleaf/tex_snippets/payoff.tex",
              Tabular("lcc"),
              [Rule(:top),
               ["", "Common interest games", "Competing interest games"],
               Rule(:mid),
               Rule(),           # a nice \hline to make it ugly
               res_mat,
               vec(gen_row("payoff")), # ragged!
               Rule(),
               vec(gen_row("max")),
               Rule(:bottom)])


#######################################################
#%% Give the same table for only the comparison games
######################################################
function vectorin(a, b)
    bset = Set(b)
    [i in bset for i in a]
end

comp_payoff_pos_df = @where(payoff_pos_df, vectorin(:round, comparison_idx))
comp_payoff_neg_df = @where(payoff_neg_df, vectorin(:round, comparison_idx))



long_pos_payoff_df = melt(comp_payoff_pos_df, [:round])

mean_pos_df = by(long_pos_payoff_df, :variable, common_payoff = :value => mean)

occursin.("avg", string.(mean_pos_df.variable))
avg_res_pos = @where(mean_pos_df, occursin.("avg", string.(:variable)))

long_neg_payoff_df = melt(comp_payoff_neg_df, [:round])

mean_neg_df = by(long_neg_payoff_df, :variable, competitive_payoff = :value => mean)

occursin.("avg", string.(mean_neg_df.variable))
avg_res_neg = @where(mean_neg_df, occursin.("avg", string.(:variable)))

perf_both_df = join(avg_res_pos, avg_res_neg, on=:variable)


perf_both_df.variable = string.(perf_both_df.variable)

@byrow! perf_both_df begin
    :variable = replace(:variable, "avg_" => "")
    :variable = replace(:variable, "_" => " ")
end

name_list = filter(x -> !(x in ["max", "payoff"]), perf_both_df.variable)
@where(perf_both_df, :variable .== name_list[1]).common_payoff[1]

#add actual pos and neg play on the other population
pos_on_pos_com = mean([payoff(pilot_pos_row_plays[i], pilot_pos_col_plays[i], pilot_neg_games[i]) for i in comparison_idx])
pos_on_neg_com = mean([payoff(pilot_pos_row_plays[i], pilot_neg_col_plays[i], pilot_neg_games[i]) for i in comparison_idx])
neg_on_pos_com = mean([payoff(pilot_neg_row_plays[i], pilot_pos_col_plays[i], pilot_neg_games[i]) for i in comparison_idx])
neg_on_neg_com = mean([payoff(pilot_neg_row_plays[i], pilot_neg_col_plays[i], pilot_neg_games[i]) for i in comparison_idx])

push!(perf_both_df, Dict(:variable => "Pos play", :common_payoff => pos_on_pos_com, :competitive_payoff => pos_on_neg_com))
push!(perf_both_df, Dict(:variable => "Neg play", :common_payoff => neg_on_pos_com, :competitive_payoff => neg_on_neg_com))


res_mat = reshape(gen_row(name_list[1]), (1,3))
for namn in name_list[2:end]
    res_mat = vcat(res_mat, gen_row(namn))
end
println(res_mat)

latex_tabular("./../overleaf/tex_snippets/comp_games_payoff.tex",
              Tabular("lcc"),
              [Rule(:top),
               ["", "Common Interst (Comparison Games) ", "Competing Interest (Comparsion Games)"],
               Rule(:mid),
               Rule(),           # a nice \hline to make it ugly
               res_mat,
               vec(gen_row("Pos play")), # ragged!
               vec(gen_row("Neg play")), # ragged!
               Rule(),
               vec(gen_row("max")),
               Rule(:bottom)])


#######################################################
#%% Fit share of different heuristics over time
#######################################################

fit_costs = Costs(0., 0., 0., 10.)
ind_pos_mh = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0.]);
ind_neg_mh = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0.]);


row_h_dist_pos = DataFrame()
for i in 1:50
    mh = deepcopy(ind_pos_mh)
    games = pilot_pos_games[[i, i+50]]
    mh = fit_prior!(mh, pilot_pos_games[[i, i+50]], pos_actual_h, pos_actual_h, fit_costs)
    h_dists = [h_distribution(mh, g, pos_actual_h, fit_costs) for g in games]
    a,b,c,d,e = round.(mean(h_dists), digits=2)
    res_dict = Dict(:round => i, :joint => a, :rowmax => b, :rowmean => c, :rowmin => d, :sim => e)
    if i == 1
        row_h_dist_pos = DataFrame(res_dict)
    else
        push!(row_h_dist_pos, res_dict)
    end
end

row_h_dist_neg = DataFrame()
for i in 1:50
    mh = deepcopy(ind_neg_mh)
    games = pilot_neg_games[[i, i+50]]
    mh = fit_prior!(mh, pilot_neg_games[[i, i+50]], neg_actual_h, neg_actual_h, fit_costs)
    h_dists = [h_distribution(mh, g, neg_actual_h, fit_costs) for g in games]
    a,b,c,d,e = round.(mean(h_dists), digits=2)
    res_dict = Dict(:round => i, :joint => a, :rowmax => b, :rowmean => c, :rowmin => d, :sim => e)
    if i == 1
        row_h_dist_neg = DataFrame(res_dict)
    else
        push!(row_h_dist_neg, res_dict)
    end
end


@with(row_h_dist_pos, plot(:round, [runmean(:joint, 10), runmean(:rowmax, 10), runmean(:rowmean, 10), runmean(:rowmin, 10), runmean(:sim, 10)], ylims=(0,1), labels=["Joint", "Row max", "Row mean", "Row min", "Sim"]))
savefig("../overleaf/figs/hdist_pos_rolling_10.png")
@with(row_h_dist_neg, plot(:round, [runmean(:joint, 10), runmean(:rowmax, 10), runmean(:rowmean, 10), runmean(:rowmin, 10), runmean(:sim, 10)], ylims=(0,1), labels=["Joint", "Row max", "Row mean", "Row min", "Sim"]))
savefig("../overleaf/figs/hdist_neg_rolling_10.png")


@df row_h_dist_pos plot(:round, [:joint, :rowmax, :rowmean, :rowmin, :sim], legend=:topleft)

#%% Predictive power of each heuristic

joint_h = JointMax(1.)
row_high = RowγHeuristic(2., 1.)
row_mean = RowγHeuristic(0., 1.)
row_min = RowγHeuristic(-2., 1.)
sim_h = SimHeuristic([RowHeuristic(0., 1.), RowMean(1.)])




row_h_loss_pos = DataFrame()
for i in 1:50
    games = pilot_pos_games[[i, i+50]]
    min_l = min_loss(pilot_pos_data[[i, i+50]])
    # joint_h = fit_h!(joint_h, pilot_pos_games[[i, i+50]], pos_actual_h)
    joint_h_loss = prediction_loss(joint_h, pilot_pos_games[[i, i+50]], pos_actual_h) - min_l
    # row_high = fit_h!(row_high, pilot_pos_games[[i, i+50]], pos_actual_h)
    row_high_loss = prediction_loss(row_high, pilot_pos_games[[i, i+50]], pos_actual_h) - min_l
    # row_mean = fit_h!(row_mean, pilot_pos_games[[i, i+50]], pos_actual_h)
    row_mean_loss = prediction_loss(row_mean, pilot_pos_games[[i, i+50]], pos_actual_h) - min_l
    # row_min = fit_h!(row_min, pilot_pos_games[[i, i+50]], pos_actual_h)
    row_min_loss = prediction_loss(row_min, pilot_pos_games[[i, i+50]], pos_actual_h) - min_l
    # sim_h = fit_h!(sim_h, pilot_pos_games[[i, i+50]], pos_actual_h)
    sim_h_loss = prediction_loss(sim_h, pilot_pos_games[[i, i+50]], pos_actual_h) - min_l
    res_dict = Dict(:round => i, :joint => joint_h_loss, :rowmax => row_high_loss, :rowmean => row_mean_loss, :rowmin => row_min_loss, :sim => sim_h_loss)
    if i == 1
        row_h_loss_pos = DataFrame(res_dict)
    else
        push!(row_h_loss_pos, res_dict)
    end
end

joint_h = fit_h!(joint_h, pilot_neg_games, neg_actual_h)
row_high = fit_h!(row_high, pilot_neg_games, neg_actual_h)
row_mean = fit_h!(row_mean, pilot_neg_games, neg_actual_h)
row_min = fit_h!(row_min, pilot_neg_games, neg_actual_h)
sim_h = fit_h!(sim_h, pilot_neg_games, neg_actual_h)

row_h_loss_neg = DataFrame()
min_loss(pilot_neg_data[[i, i+50]])
for i in 1:50
    games = pilot_neg_games[[i, i+50]]
    min_l = min_loss(pilot_neg_data[[i, i+50]])
    # joint_h = fit_h!(joint_h, pilot_neg_games[[i, i+50]], neg_actual_h)
    joint_h_loss = prediction_loss(joint_h, pilot_neg_games[[i, i+50]], neg_actual_h) - min_l
    # row_high = fit_h!(row_high, pilot_neg_games[[i, i+50]], neg_actual_h)
    row_high_loss = prediction_loss(row_high, pilot_neg_games[[i, i+50]], neg_actual_h) - min_l
    # row_mean = fit_h!(row_mean, pilot_neg_games[[i, i+50]], neg_actual_h)
    row_mean_loss = prediction_loss(row_mean, pilot_neg_games[[i, i+50]], neg_actual_h) - min_l
    # row_min = fit_h!(row_min, pilot_neg_games[[i, i+50]], neg_actual_h)
    row_min_loss = prediction_loss(row_min, pilot_neg_games[[i, i+50]], neg_actual_h) - min_l
    # sim_h = fit_h!(sim_h, pilot_neg_games[[i, i+50]], neg_actual_h)
    sim_h_loss = prediction_loss(sim_h, pilot_neg_games[[i, i+50]], neg_actual_h) - min_l
    res_dict = Dict(:round => i, :joint => joint_h_loss, :rowmax => row_high_loss, :rowmean => row_mean_loss, :rowmin => row_min_loss, :sim => sim_h_loss)
    if i == 1
        row_h_loss_neg = DataFrame(res_dict)
    else
        push!(row_h_loss_neg, res_dict)
    end
end


gr()
@with(row_h_loss_pos, plot(:round, [runmean(:joint, 10), runmean(:rowmax, 10), runmean(:rowmean, 10), runmean(:rowmin, 10), runmean(:sim, 10)], ylims=(0,2), labels=["Joint", "Row max", "Row mean", "Row min", "Sim"]))
@with(row_h_loss_pos, plot(:round, [runmean(:joint, 1), runmean(:rowmax, 1), runmean(:rowmean, 1), runmean(:rowmin, 1), runmean(:sim, 1)], ylims=(0,2), labels=["Joint", "Row max", "Row mean", "Row min", "Sim"]))

@with(row_h_loss_neg, plot(:round, [runmean(:joint, 10), runmean(:rowmax, 10), runmean(:rowmean, 10), runmean(:rowmin, 10), runmean(:sim, 10)], labels=["Joint", "Row max", "Row mean", "Row min", "Sim"]))
@with(row_h_loss_neg, plot(:round, [runmean(:joint, 5), runmean(:rowmax, 5), runmean(:rowmean, 5), runmean(:rowmin, 5), runmean(:sim, 5)], labels=["Joint", "Row max", "Row mean", "Row min", "Sim"]))



@df row_h_dist_pos plot(:round, [:joint, :rowmax, :rowmean, :rowmin, :sim], legend=:topleft)



#%% Estimate γ over time

row_γ_dist_pos = DataFrame()
row_h = RowHeuristic(0., 2.)
for i in 1:50
    h = deepcopy(row_h)
    games = pilot_pos_games[[i, i+50]]
    h = fit_h!(h, pilot_pos_games[[i, i+50]], pos_actual_h)
    res_dict = Dict(:round => i, :γ => clamp(h.γ,-3,3), :λ => clamp(h.λ, 0,5))
    if i == 1
        row_γ_dist_pos = DataFrame(res_dict)
    else
        push!(row_γ_dist_pos, res_dict)
    end
end

row_γ_dist_neg = DataFrame()
row_h = RowHeuristic(0., 2.)
for i in 1:50
    h = deepcopy(row_h)
    games = pilot_neg_games[[i, i+50]]
    h = fit_h!(h, pilot_neg_games[[i, i+50]], neg_actual_h)
    res_dict = Dict(:round => i, :γ => clamp(h.γ,-3,3), :λ => clamp(h.λ, 0,5))
    if i == 1
        row_γ_dist_neg = DataFrame(res_dict)
    else
        push!(row_γ_dist_neg, res_dict)
    end
end

@with(row_γ_dist_pos, plot(:round, runmean(:γ, 10), labels=["Gamma"]))
savefig("../overleaf/figs/gamm_dist_pos_rolling_10.png")
@with(row_γ_dist_neg, plot(:round, runmean(:γ, 10), labels=["Gamma"]))
savefig("../overleaf/figs/gamm_dist_neg_rolling_10.png")





#%% Print example games
function gen_tabular_tex_game(game, linspace=false)
    str = """\\begin{tabular}{|c|c|c|}"""
    if linspace
        str = str*"\\addlinespace[0.5cm]"
    end
    str = str*"""\\hline """
    for i in 1:3
        str = str*"""
        \$$(Int(game.row[i,1])), $(Int(game.col[i,1]))\$ & \$$(Int(game.row[i,2])), $(Int(game.col[i,2]))\$ & \$$(Int(game.row[i,3])), $(Int(game.col[i,3]))\$ \\\\ \\hline """
    end
    str = str*"\\end{tabular}"
end

function gen_game_table(games, idxs)
    res = " "
    for (i, (game, idx)) in enumerate(zip(games[idxs],idxs))
        res = res*"\\parbox{.45\\linewidth}{\\centering "
        if i > 2
            res = res*gen_tabular_tex_game(game, true)
            # res = res*gen_tabular_tex_game(games[idx], true)
        else
            res = res*gen_tabular_tex_game(game)
            # res = res*gen_tabular_tex_game(games[idx])
        end
        res = res*" \\caption*{Game $idx} }"
        if i != length(idxs)
            res = res*" \\hfill "
        end
    end
    return res
end

gen_game_table(pilot_pos_games, [1,2,3,4])

open("../overleaf/tex_snippets/pos_games.tex","w") do f
    write(f, gen_game_table(pilot_pos_games, [1,2,3,4]))
end

open("../overleaf/tex_snippets/neg_games.tex","w") do f
    write(f, gen_game_table(pilot_neg_games, [1,2,3,4]))
end


for (i, (game, idx)) in enumerate(zip(games[idxs],idxs))
    println(game)
end
