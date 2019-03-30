using Flux
using Flux: @epochs
using Flux.Tracker
using Flux.Tracker: grad, update!
using LinearAlgebra: norm
include("Heuristics.jl")
include("DeepLayers.jl")
# push!(LOAD_PATH, pwd())
# using DeepLayers

opp_h = QLK(0.2, 0.1, 3.)
opp_h2 = QLK(0.25, 0.1, 3.)
g = Game(3,0.1)
play_distribution(opp_h, g)

games = [Game(3,0.1) for i in 1:1000];
test_games = [Game(3,0.1) for i in 1:1000];

model = Chain(Game_Dense(1, 10, sigmoid), Game_Dense(10,10), Game_Soft(10), Action_Response(1), Action_Response(2), Last(3))
data = [(g, play_distribution(opp_h, g)) for g in games]
test_data = [(g, play_distribution(opp_h, g)) for g in test_games]

loss(x::Game, y) = Flux.crossentropy(model(x), y) + 0.0001*sum(norm, params(model))
loss_no_norm(x::Game, y) = Flux.crossentropy(model(x), y)
loss(x::Vector{Float64}, y) = Flux.crossentropy(x, y)
loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(g,y) for (g,y) in data])
loss_no_norm(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss_no_norm(g,y) for (g,y) in data])
min_loss(data::Array{Tuple{Game,Array{Float64,1}},1}) = sum([loss(y,y) for (g,y) in data])

model(data[3][1])
compare_loos = sum([loss(play_distribution(opp_h2, g), play_distribution(opp_h2, g)) for g in games])

min_loss(data)
loss(data)
loss_no_norm(data)

H1, H2, U1, U2 = Game_Dense(1,10)(data[3][1])

i = 1
H1[1,2]
k = 2

H = ones(3,3,10)
H[1,2,:]


HCmax = [[maximum([H1[i,l][k] for l in 1:size(H1)[2]]) for k in 1:length(H1[1,1])] for i in 1:size(H1)[1], j in 1:size(H1)[2]]
HRmax = [[maximum([H1[l,i][k] for l in 1:size(H1)[1]]) for k in 1:length(H1[1,1])] for i in 1:size(H1)[1], j in 1:size(H1)[2]]

[Hmax[i,j][1] for i in 1:3, j in 1:3]

ps = Flux.params(model)

opt = ADAM(0.001, (0.9, 0.999))

evalcb() = @show(loss_no_norm(data), loss(data), min_loss(data))

@epochs 5 Flux.train!(loss_no_norm, ps, data, opt, cb = Flux.throttle(evalcb,5))






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
