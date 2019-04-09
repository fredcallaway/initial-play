using Flux
using Flux: @epochs
using Flux.Tracker
using Flux.Tracker: grad, update!
using LinearAlgebra: norm
using StatsBase: entropy
using JSON
using CSV
using DataFrames
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
positive_games_df = CSV.read("pilot/dataframes/positive_games_df.csv")
negative_games_df = CSV.read("pilot/dataframes/negative_games_df.csv")


comparison_idx = [31, 37, 41, 44, 49]
#%% Look at comparison games

for i in comparison_idx
    game = json_to_game(first(positive_games_df[positive_games_df.round .== i, :row]))
    println(game)
    pos_row_play = pos_dfrow = first(positive_games_df[positive_games_df.round .== i, :row_play])
    pos_col_play = pos_dfrow = first(positive_games_df[positive_games_df.round .== i, :col_play])
    println("pos row: ", pos_row_play)
    println("pos col: ", pos_col_play)
    neg_row_play = neg_dfrow = first(negative_games_df[negative_games_df.round .== i, :row_play])
    neg_col_play = neg_dfrow = first(negative_games_df[negative_games_df.round .== i, :col_play])
    println("neg row: ", neg_row_play)
    println("neg col: ", neg_col_play)
    println("------------------------------------------------------")
end


#%% General loss functions
loss(x::Vector{Float64}, y) = isnan(Flux.crossentropy(x, y)) ? Flux.crossentropy((x .+ 0.01)./1.03, (y .+ 0.01)./1.03) : Flux.crossentropy(x, y)
rand_loss(y) = Flux.crossentropy(ones(length(y))/length(y), y)

loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])/length(data)
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])/length(data)
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])/length(data)
rand_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([rand_loss(y) for (g,y) in data])/length(data)



###########################################################################
#%% Positive Treatment: Load the data
###########################################################################
comparison_idx = [31, 37, 41, 44, 49]
later_com = 50 .+ comparison_idx
comparison_idx = [comparison_idx..., later_com...]

pilot_pos_data = [(json_to_game(first(positive_games_df[positive_games_df.round .== i, :row])), convert(Vector{Float64}, JSON.parse(first(positive_games_df[positive_games_df.round .== i, :row_play])))) for i in 1:50];
append!(pilot_pos_data ,[(json_to_game(first(positive_games_df[positive_games_df.round .== i, :col])), convert(Vector{Float64}, JSON.parse(first(positive_games_df[positive_games_df.round .== i, :col_play])))) for i in 1:50]);

# By adding a minor random noise to the only symmetric game we can distinguish the game from the row and columns perspective
pilot_pos_data[41] = (Game(pilot_pos_data[41][1].row .+ rand(3,3)*0.01, pilot_pos_data[41][1].col .+ rand(3,3)*0.01), pilot_pos_data[41][2])
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
#%% Positive Treatment: QCH
####################################################
pos_actual_h = CacheHeuristic(pilot_pos_games, pilot_pos_row_plays);
# pilot_pos_opp_h = CacheHeuristic(transpose.(pilot_pos_games), pilot_pos_col_plays)
pos_qch_h = QCH(0.07, 0.64, 1.5, 1.7, 1.9)
best_fit_qch_pos = fit_h!(pos_qch_h, pilot_pos_games, pos_actual_h)
# best_fit_qch_pos = fit_h!(pos_qch_h, pilot_pos_games[comparison_idx], pos_actual_h)
# prediction_loss(pos_qch_h, pilot_pos_games[comparison_idx], pos_actual_h)
prediction_loss(pos_qch_h, pilot_pos_games, pos_actual_h)

best_fit_qch_pos = fit_h!(pos_qch_h, pilot_pos_train_games, pos_actual_h)
@show prediction_loss(pos_qch_h, pilot_pos_train_games, pos_actual_h);
@show prediction_loss(pos_qch_h, pilot_pos_test_games, pos_actual_h);
@show prediction_loss(pos_qch_h, pilot_pos_games[comparison_idx], pos_actual_h)


####################################################
#%% Positive Treatment: MetaHeuristic
####################################################
# mh_pos = MetaHeuristic([JointMax(3.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);
mh_pos = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
n_fit_iter = 5
# costs = Costs(0.08, 0.15, 0.07, 1.5)
costs = Costs(0.1, 0.1, 0.2, 1.5)

fit_mh_pos = deepcopy(mh_pos)
for i in 1:n_fit_iter
    fit_mh_pos = fit_prior!(fit_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs)
    fit_mh_pos = fit_h!(fit_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs; init_x = get_parameters(fit_mh_pos))
end

prediction_loss(fit_mh_pos, pilot_pos_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_data)
prediction_loss(fit_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs)
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

prediction_loss(opt_mh_pos, pilot_pos_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_data)
prediction_loss(opt_mh_pos, pilot_pos_train_games, pos_actual_h, pos_actual_h, costs)
prediction_loss(opt_mh_pos, pilot_pos_test_games, pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_test_data)
prediction_loss(opt_mh_pos, pilot_pos_games[comparison_idx], pos_actual_h, pos_actual_h, costs)
min_loss(pilot_pos_data[comparison_idx])
h_dists = [h_distribution(opt_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_h_dist = mean(h_dists)





####################################################
#%% Positive Treatment: Setup and run Deep Heuristic without action response
####################################################
γ = 0.001 # Overfitting penalty
# model_0_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
model_0_pos = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Last(1))
# model_0_pos = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

# Specific loss function for model_0_pos
loss(x::Game, y) = Flux.crossentropy(model_0_pos(x), y) + γ*sum(norm, params(model_0_pos))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_0_pos(x), y)

#%% estimate
ps = Flux.params(model_0_pos)
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(rand_loss(pilot_pos_train_data), loss_no_norm(pilot_pos_train_data), min_loss(pilot_pos_train_data), rand_loss(pilot_pos_test_data), loss_no_norm(pilot_pos_test_data), min_loss(pilot_pos_test_data),loss_no_norm(pilot_pos_data[comparison_idx]))
println("clear print are")
5
@epochs 50 Flux.train!(loss, ps, pilot_pos_train_data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% Positive Treatment: Add action response layers
####################################################
model_action_pos = deepcopy(Chain(model_0_pos.layers[1:(length(model_0_pos.layers)-1)]..., Action_Response(1), Last(2)))

loss_no_norm(pilot_pos_data[comparison_idx])
loss_no_norm(pilot_pos_data)
loss_no_norm(pilot_pos_test_data)
#%% Estimate with action response layers
loss(x::Game, y) = Flux.crossentropy(model_action_pos(x), y) + γ*sum(norm, params(model_action_pos))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_action_pos(x), y)
ps = Flux.params(model_action_pos)
opt = ADAM(0.001, (0.9, 0.999))
5
evalcb() = @show(rand_loss(pilot_pos_train_data), loss_no_norm(pilot_pos_train_data), min_loss(pilot_pos_train_data), rand_loss(pilot_pos_test_data), loss_no_norm(pilot_pos_test_data), min_loss(pilot_pos_test_data), loss_no_norm(pilot_pos_data[comparison_idx]))
println("apa")
4
@epochs 50 Flux.train!(loss, ps, pilot_pos_train_data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% Positive Treatment: Setup and run OPTIMAL Deep Heuristic without action response
####################################################
γ = 0.001 # Overfitting penalty
sim_cost = 0.1
exact_cost = 0.5
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
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(loss(pilot_pos_data), loss_no_cost(pilot_pos_data), rand_loss(pilot_pos_data), loss_no_norm(pilot_pos_data), min_loss(pilot_pos_data), loss_no_norm(pilot_pos_data[comparison_idx]))
println("clear print are")
5
@epochs 50 Flux.train!(loss, ps, pilot_pos_data, opt, cb = Flux.throttle(evalcb,5))


###########################################################################
#%% Negative Treatment: Load the data
###########################################################################
comparison_idx = [31, 37, 41, 44, 49]
later_com = 50 .+ comparison_idx
comparison_idx = [comparison_idx..., later_com...]

pilot_neg_data = [(json_to_game(first(negative_games_df[negative_games_df.round .== i, :row])), convert(Vector{Float64}, JSON.parse(first(negative_games_df[negative_games_df.round .== i, :row_play])))) for i in 1:50];
append!(pilot_neg_data ,[(json_to_game(first(negative_games_df[negative_games_df.round .== i, :col])), convert(Vector{Float64}, JSON.parse(first(negative_games_df[negative_games_df.round .== i, :col_play])))) for i in 1:50]);

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
#%% Negative Treatment: QCH
####################################################
neg_actual_h = CacheHeuristic(pilot_neg_games, pilot_neg_row_plays)
# pilot_neg_opp_h = CacheHeuristic(transnege.(pilot_neg_games), pilot_neg_col_plays)
neg_qch_h = QCH(0.07, 0.64, 1.5, 1.7, 1.9)
best_fit_qch_neg = fit_h!(neg_qch_h, pilot_neg_games, neg_actual_h)
# best_fit_qch_neg = fit_h!(neg_qch_h, pilot_neg_games[comparison_idx], neg_actual_h)
# prediction_loss(neg_qch_h, pilot_neg_games[comparison_idx], neg_actual_h)
prediction_loss(neg_qch_h, pilot_neg_games, neg_actual_h)

best_fit_qch_neg = fit_h!(neg_qch_h, pilot_neg_train_games, neg_actual_h)
prediction_loss(neg_qch_h, pilot_neg_train_games, neg_actual_h)
min_loss(pilot_neg_train_data)
prediction_loss(neg_qch_h, pilot_neg_test_games, neg_actual_h)
min_loss(pilot_neg_test_data)
prediction_loss(neg_qch_h, pilot_neg_games[comparison_idx], neg_actual_h)
min_loss(pilot_neg_data[comparison_idx])


####################################################
#%% Negative Treatment: MetaHeuristic
####################################################
# mh_neg = MetaHeuristic([JointMax(3.), RowγHeuristic(3., 2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0., 0., 0., 0.]);
mh_neg = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);

# costs = Costs(0.03, 0.15, 0.3, 1.5)
# costs = Costs(0.1, 0.1, 0.3, 1.5)
costs = Costs(0.1, 0.1, 0.2, 1.5)


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

#%% Compare positive and negative
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
#%% Negative Treatment: Setup and run Deep Heuristic without action response
####################################################
γ = 0.001 # Overfitting penalty
# model_0_neg = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50, sigmoid), Game_Dense(50,30), Game_Soft(30), Last(1))
model_0_neg = Chain(Game_Dense_full(1, 100, sigmoid), Game_Dense(100,50), Game_Soft(50), Last(1))
# model_0_neg = Chain(Game_Dense(1, 30, sigmoid), Game_Dense(30, 30, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3))

# Specific loss function for model_0_neg
loss(x::Game, y) = Flux.crossentropy(model_0_neg(x), y) + γ*sum(norm, params(model_0_neg))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_0_neg(x), y)

#%% estimate
ps = Flux.params(model_0_neg)
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(rand_loss(pilot_neg_train_data), loss_no_norm(pilot_neg_train_data), min_loss(pilot_neg_train_data), rand_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_test_data), min_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_data[comparison_idx]), loss_no_norm(pilot_pos_data))
println("clear print are")
5
@epochs 50 Flux.train!(loss, ps, pilot_neg_train_data, opt, cb = Flux.throttle(evalcb,5))


####################################################
#%% Negative Treatment: Add action response layers
####################################################
model_action_neg = deepcopy(Chain(model_0_neg.layers[1:(length(model_0_neg.layers)-1)]..., Action_Response(1), Last(2)))
loss(x::Game, y) = Flux.crossentropy(model_action_neg(x), y) + γ*sum(norm, params(model_action_neg))
loss_no_norm(x::Game, y) = Flux.crossentropy(model_action_neg(x), y)

loss_no_norm(pilot_neg_data[comparison_idx])
loss_no_norm(pilot_neg_data)
#%% Estimate with action response layers
ps = Flux.params(model_action_neg)
opt = ADAM(0.0001, (0.9, 0.999))
evalcb() = @show(rand_loss(pilot_neg_train_data), loss_no_norm(pilot_neg_train_data), min_loss(pilot_neg_train_data), rand_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_test_data), min_loss(pilot_neg_test_data), loss_no_norm(pilot_neg_data[comparison_idx]),loss_no_norm(pilot_pos_data))
println("apa")
4
@epochs 50 Flux.train!(loss, ps, pilot_neg_train_data, opt, cb = Flux.throttle(evalcb,5))

####################################################
#%% Negative Treatment: Setup and run OPTIMAL Deep Heuristic without action response
####################################################
γ = 0.001 # Overfitting penalty
sim_cost = 0.1
exact_cost = 0.7
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
opt = ADAM(0.001, (0.9, 0.999))
evalcb() = @show(loss(pilot_neg_data), loss_no_cost(pilot_neg_data), rand_loss(pilot_neg_data), loss_no_norm(pilot_neg_data), min_loss(pilot_neg_data), loss_no_norm(pilot_neg_data[comparison_idx]), loss_no_norm(pilot_pos_data))
println("clear print are")
5
@epochs 50 Flux.train!(loss, ps, pilot_neg_train_data, opt, cb = Flux.throttle(evalcb,5))


######################################################
#%% Compare the estimated heuristics
######################################################

println("------- QCH ----------")
println("Neg QCH: ", best_fit_qch_neg)
println("Pos QCH: ", best_fit_qch_pos)
println("neg on neg: ", prediction_loss(best_fit_qch_neg, pilot_neg_games, neg_actual_h))
println("pos on neg: ", prediction_loss(best_fit_qch_pos, pilot_neg_games, neg_actual_h))
println("neg on neg comp: ", prediction_loss(best_fit_qch_neg, pilot_neg_games[comparison_idx], neg_actual_h))
println("pos on neg comp: ", prediction_loss(best_fit_qch_pos, pilot_neg_games[comparison_idx], neg_actual_h))


println("neg on pos: ", prediction_loss(best_fit_qch_neg, pilot_pos_games, pos_actual_h))
println("pos on pos: ", prediction_loss(best_fit_qch_pos, pilot_pos_games, pos_actual_h))
println("neg on pos comp: ", prediction_loss(best_fit_qch_neg, pilot_pos_games[comparison_idx], pos_actual_h))
println("pos on pos comp: ", prediction_loss(best_fit_qch_pos, pilot_pos_games[comparison_idx], pos_actual_h))

println("------- Meta Heuristics ----------")
println("Neg MH: ", fit_mh_neg)
h_dists_neg = [h_distribution(fit_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games];
avg_h_dist_neg = mean(h_dists_neg);
println("Avg h neg: ", avg_h_dist_neg)
println("Pos MH: ", fit_mh_pos)
h_dists_pos = [h_distribution(fit_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_h_dist_pos = mean(h_dists_pos);
println("Avg h pos: ", avg_h_dist_pos)
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
h_dists_neg = [h_distribution(opt_mh_neg, g, neg_actual_h, costs) for g in pilot_neg_games];
avg_h_dist_neg = mean(h_dists_neg);
println("Avg h neg: ", avg_h_dist_neg)
println("Pos MH: ", opt_mh_pos)
h_dists_pos = [h_distribution(opt_mh_pos, g, pos_actual_h, costs) for g in pilot_pos_games];
avg_h_dist_pos = mean(h_dists_pos);
println("Avg h pos: ", avg_h_dist_pos)
println("neg on neg: ", prediction_loss(opt_mh_neg, pilot_neg_games, neg_actual_h))
println("pos on neg: ", prediction_loss(opt_mh_pos, pilot_neg_games, neg_actual_h))
println("neg on neg comp: ", prediction_loss(opt_mh_neg, pilot_neg_games[comparison_idx], neg_actual_h))
println("pos on neg comp: ", prediction_loss(opt_mh_pos, pilot_neg_games[comparison_idx], neg_actual_h))

println("neg on pos: ", prediction_loss(opt_mh_neg, pilot_pos_games, pos_actual_h))
println("pos on pos: ", prediction_loss(opt_mh_pos, pilot_pos_games, pos_actual_h))
println("neg on pos comp: ", prediction_loss(opt_mh_neg, pilot_pos_games[comparison_idx], pos_actual_h))
println("pos on pos comp: ", prediction_loss(opt_mh_pos, pilot_pos_games[comparison_idx], pos_actual_h))

println("------- Deep Heuristics ----------")
# loss_no_norm_neg(x::Game, y) = Flux.crossentropy(model_action_neg(x), y);
loss_no_norm_neg(x::Game, y) = Flux.crossentropy(model_0_neg(x), y);
loss_no_norm_neg(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm_neg(g,y) for (g,y) in data])/length(data);

# loss_no_norm_pos(x::Game, y) = Flux.crossentropy(model_action_pos(x), y);
loss_no_norm_pos(x::Game, y) = Flux.crossentropy(model_0_pos(x), y);
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
#%% Generate Data Frame
######################################################
data_dict = Dict()
data_dict["negative"] = Dict(
"all" => Dict("data" => pilot_neg_data, "games" => pilot_neg_games, "row_play" => pilot_neg_row_plays),
"train" => Dict("data" => pilot_neg_train_data, "games" => pilot_neg_train_games, "row_play" => pilot_neg_train_row), "test" => Dict("data" => pilot_neg_test_data, "games" => pilot_neg_test_games, "row_play" => pilot_neg_test_row),
"comparison" => Dict("data" => pilot_neg_data[comparison_idx], "games" => pilot_neg_games[comparison_idx], "row_play" => pilot_neg_row_plays[comparison_idx]),
"actual_h" => neg_actual_h);

data_dict["positive"] = Dict(
"all" => Dict("data" => pilot_pos_data, "games" => pilot_pos_games, "row_play" => pilot_pos_row_plays),
"train" => Dict("data" => pilot_pos_train_data, "games" => pilot_pos_train_games, "row_play" => pilot_pos_train_row), "test" => Dict("data" => pilot_pos_test_data, "games" => pilot_pos_test_games, "row_play" => pilot_pos_test_row),
"comparison" => Dict("data" => pilot_pos_data[comparison_idx], "games" => pilot_pos_games[comparison_idx], "row_play" => pilot_pos_row_plays[comparison_idx]),
"actual_h" => pos_actual_h);


res_df = DataFrame()

for treatment in ["negative", "positive"]
    for data_type in ["all", "train", "test", "comparison"]
        res_dict = Dict{Any, Any}(:data_type => data_type, :treatment => treatment)
        data = data_dict[treatment][data_type]
        actual = data_dict[treatment]["actual_h"]
        res_dict[:random] = rand_loss(data["data"])
        res_dict[:minimum] = min_loss(data["data"])
        res_dict[:pos_QCH] = prediction_loss(best_fit_qch_pos, data["games"], actual)
        res_dict[:neg_QCH] = prediction_loss(best_fit_qch_neg, data["games"], actual)
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


data_names = [:random, :neg_QCH, :pos_QCH, :fit_mh_neg, :fit_mh_pos, :opt_mh_neg, :opt_mh_pos, :fit_deep_neg,  :fit_deep_pos, :opt_deep_neg, :opt_deep_pos, :minimum]

treat = "negative"
data_type = "all"
vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), data_names]))
ctg = [repeat(["negative"], 5)..., repeat(["positive"], 5)...]
nam = [repeat(["QCH", "fit mh", "opt mh", "fit deep", "opt deep"],2)...]
bar_vals = hcat(transpose(reshape(vals[2:(end-1)], (2,5))))
plt = groupedbar(nam, bar_vals, group=ctg, lw=1, framestyle = :box, title = treat*"-"*data_type, ylims=(0,1.5))
plot!([vals[1]], linetype=:hline, width=2, label="random loss", color=:grey)
plot!([vals[12]], linetype=:hline, width=2, label="min loss", color=:black)
hline(plt; vals=0.3)


plots_vec = []
for data_type in ["all", "train", "test", "comparison"], treat in ["negative", "positive"]
    vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), data_names]))
    ctg = [repeat(["negative"], 5)..., repeat(["positive"], 5)...]
    nam = [repeat(["QCH", "fit mh", "opt mh", "fit deep", "opt deep"],2)...]
    bar_vals = hcat(transpose(reshape(vals[2:(end-1)], (2,5))))
    plt = groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*"-"*data_type, ylims=(0,1.3))
    plot!([vals[1]], linetype=:hline, width=2, label="random loss", color=:grey)
    plot!([vals[12]], linetype=:hline, width=2, label="min loss", color=:black)
    push!(plots_vec, plt)
end
#
#
# for data_type in ["all", "train", "test", "comparison"], treat in ["negative", "positive"]
#     vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), data_names]))
#     ctg = [repeat(["minimum"], 5)..., repeat(["negative"], 5)..., repeat(["positive"], 5)..., repeat(["random"], 5)...]
#     nam = [repeat(["QCH", "fit mh", "opt mh", "fit deep", "opt deep"],4)...]
#     bar_vals = hcat(repeat([vals[12]],5), transpose(reshape(vals[2:(end-1)], (2,5))), repeat([vals[1]],5))
#     push!(plots_vec, groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*"-"*data_type))
# end

length(plots_vec)
plot(plots_vec..., layout=(4,2), size=(1191,1684))

savefig("test.png")

res_df
