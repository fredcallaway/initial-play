using Flux
using Flux: @epochs
using Flux.Tracker
using Flux.Tracker: grad, update!
using LinearAlgebra: norm
include("Heuristics.jl")
include("DeepLayers.jl")


opp_h = QLK(0.2, 0.1, 3.)
opp_h2 = QLK(0.25, 0.1, 3.)

games = [Game(3,0.1) for i in 1:100];
test_games = [Game(3,0.1) for i in 1:100];

data = [(g, play_distribution(opp_h, g)) for g in games];
test_data = [(g, play_distribution(opp_h, g)) for g in test_games];

model = Chain(Game_Dense_full(1, 30, sigmoid), Game_Dense(30,10), Game_Soft(10), Action_Response(1), Action_Response(2), Last(3))

loss(x::Game, y) =Flux.crossentropy(model(x), y) + 0.001*sum(norm, params(model))
loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
loss(x::Vector{Float64}, y) = Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])

model(data[3][1])
min_loss(data)
loss(data)
loss_no_norm(data)

ps = Flux.params(model)

opt = ADAM(0.01, (0.9, 0.999))
evalcb() = @show(loss_no_norm(data), min_loss(data), loss_no_norm(test_data), min_loss(test_data))
@epochs 10 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(evalcb,5))


#%%
costs = Costs(0.03, 0.1, 0.3, 5.)
ρ = 0.9
games = [Game(3, ρ) for i in 1:100];
test_games = [Game(3, ρ) for i in 1:100];
mh = MetaHeuristic([JointMax(2.), NashHeuristic(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [30., 30., 0., 0., 0., 0., 0., 30.]);
mh = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [30., 0., 30., 0., 0., 20., 30.]);
# mh = opt_prior!(mh, games, opp_h, costs);
h_dists = [h_distribution(mh, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)
data =  [(g, play_distribution(mh, g)) for g in games];
test_data = [(g, play_distribution(mh, g)) for g in test_games];
model = Chain(Game_Dense_full(1, 50, sigmoid), Game_Dense(50,30), Game_Soft(30), Action_Response(1), Action_Response(2), Last(3));

# H1, H2, U1, U2 = (Game_Soft(10)(Game_Dense(50,10)(Game_Dense(1, 50, sigmoid)(data[1][1]))))
model(data[1][1])
min_loss(data)
loss(data)
loss_no_norm(data)
ps = Flux.params(model)
opt = ADAM(0.01, (0.9, 0.999))
# opt = Descent(0.1)
evalcb() = @show(loss_no_norm(data), min_loss(data), loss_no_norm(test_data), min_loss(test_data))
@epochs 10 Flux.train!(loss_no_norm, ps, data, opt, cb = Flux.throttle(evalcb,5))

actual_h = CacheHeuristic(games, [play_distribution(mh, g) for g in games])
actual_h_test = CacheHeuristic(test_games, [play_distribution(mh, g) for g in games])
qlk_h = QLK(0.07, 0.64, 2.3)
# best_fit_qlk = fit_h!(qlk_h, exp_games, actual_h; loss_f = mean_square)
best_fit_qlk = fit_h!(qlk_h, games, actual_h)
prediction_loss(qlk_h, games, actual_h)
prediction_loss(qlk_h, test_games, actual_h_test)



games = [Game(3, 0.8) for i in 1:1000]
mh_positive = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
mh_positive = opt_prior!(mh_positive, games, opp_h, costs)
mh_positive_no = MetaHeuristic([RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0.])
mh_positive_no = opt_prior!(mh_positive_no, games, opp_h, costs)
# mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(mh))
h_dists = [h_distribution(mh_positive, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)





a = Affine(10, 5)
sigmoid.(a(rand(10)))


model = Chain(Affine(100,50, sigmoid), Affine(50,2, sigmoid), softmax)
loss(x,y) = Flux.crossentropy(model(x),y)
ps = Flux.params(model)

x = randn(100,2)
y = [[0;1], [1;0]]

data = zip(x,y)
opt = Descent(0.1)

evalcb() = @show(loss(x, y))

@epochs 100 Flux.train!(loss, ps, data, opt, cb = evalcb)


#%%
m = Chain(
  Affine(10, 32, σ),
  Affine(32, 10), softmax)

x = [rand(10)]
ys = [rand( 10)]
data = zip(xs, ys)
data = Iterators.repeated((xs[1],ys[1]),1000)
loss(x, y) = Flux.crossentropy(m(x), y)
opt = ADAM(0.001, (0.9, 0.999))
ps = Flux.params(m)
evalcb() = @show(loss(xs[1], ys[1]))

Flux.train!(loss, ps, data, opt, cb = Flux.throttle(evalcb, 3))

Flux.onehot(:c, [:a, :b, :c])
