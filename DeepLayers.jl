using Flux
using CSV
using DataFrames
using Base
using JSON
using Distributed
using Glob
using BenchmarkTools
using SplitApplyCombine


include("Heuristics.jl")
# include("model.jl")
# include("DeepLayers_bkp.jl")


#%% So we have some data to test it on
# function load_treatment_data(treatment)
    # files = glob("data/processed/$treatment/*_play_distributions.csv")
    # data = vcat(map(load_data, files)...)
# end

# Data = Array{Tuple{Game,Array{Float64,1}},1}
# parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))


# function load_data(file)::Data
#     file
#     df = CSV.read(file);
#     row_plays = Data()
#     col_plays = Data()
#     for row in eachrow(df)
#         row_game = json_to_game(row.row_game)
#         row_game.row[1,2] += rand()*1e-7 # This can't be a constant number if we want to
#         row_game.col[1,2] += rand()*1e-7 # separate behavior in comparison games in different treatments.
#         row_play_dist = parse_play(row.row_play)
#         col_game = transpose(row_game)
#         col_play_dist = parse_play(row.col_play)
#         push!(row_plays, (row_game, row_play_dist))
#         push!(col_plays, (col_game, col_play_dist))
#     end
#     append!(row_plays, col_plays)
# end
#
# all_data = Dict(
#     :pos => load_treatment_data("positive"),
#     :neg => load_treatment_data("negative"),
# )

#%%
function gen_feats(g::Game)
    # feats = (g.row, g.col, [g.row, g.col, [min(g.row[i,j], g.col[i,j]) for i in 1:3, j in 1:3], [mapslices(x -> f(x)*ones(length(x)), u, dims=d) for f in [maximum, mean, minimum], d in [1,2], u in [g.row, g.col]]...], g.row, g.col)
    f_vals = [mapslices(x -> f(x)*ones(length(x)), u, dims=d) for f in [maximum, mean, minimum], d in [1,2], u in [g.row, g.col]]
    feats = [[g.row[i,j], g.col[i,j], min(g.row[i,j], g.col[i,j]), [f_vals[f,d,u][i,j] for f in 1:3, d in 1:2, u in 1:2]...] for i in 1:3, j in 1:3]
    # feats = (g.row, g.col, reshape(vcat(feats...), (15,3,3)))
    feats = (g.row, g.col, feats)
end

struct Game_Dense
    mᴿ::Dense
    mᶜ::Dense
end

Game_Dense(in::Integer, out::Integer, σ = identity) = begin
    mᴿ = Dense(in, out, σ)
    mᶜ = Dense(in, out, σ)
    Game_Dense(mᴿ, mᶜ)
end

(m::Game_Dense)(t) = m(t...)
(m::Game_Dense)(Uᴿ, Uᶜ, H) = m(Uᴿ, Uᶜ, H, H)
(m::Game_Dense)(Uᴿ, Uᶜ, Hᴿ, Hᶜ) = begin
    Uᴿ, Uᶜ, m.mᴿ.(Hᴿ), m.mᶜ.(Hᶜ)
end
(m::Game_Dense)(g::Game) = m(gen_feats(g)...)


struct Game_Soft{T,S}
    Wᴿ::T
    Wᶜ::S
end

Game_Soft(in::Integer) = Game_Soft(randn(in), randn(in))


(m::Game_Soft)(t) = m(t...)
(m::Game_Soft)(Uᴿ, Uᶜ, Hᴿ, Hᶜ) = begin
    Fᴿ_mat = reshape(hcat(Hᴿ...), (:,3,3))
    Sᴿ = softmax(reshape(sum(Fᴿ_mat, dims=3), (:,3));dims=2)
    wᴿ = softmax(m.Wᴿ)
    aᴿ = reshape(reshape(wᴿ,(1,:))*Sᴿ,(:))

    Fᶜ_mat = reshape(hcat(Hᶜ...), (:,3,3))
    Sᶜ = softmax(reshape(sum(Fᶜ_mat, dims=2), (:,3));dims=2)
    wᶜ = softmax(m.Wᶜ)
    aᶜ = reshape(reshape(wᶜ, (1,:))*Sᶜ, (:))

  return  Uᴿ, Uᶜ, [aᴿ], [aᶜ]
end


struct Action_Response
  vᴿ
  vᶜ
  λᴿ
  λᶜ
end

(m::Action_Response)(t) = m(t...)
Action_Response(level::Integer) = Action_Response(randn(level), randn(level), 3*rand(), 3*rand())

m = Action_Response(1)

(m::Action_Response)(Uᴿ, Uᶜ, Aᴿ, Aᶜ) =  begin
  n_rows = size(Uᴿ)[1]
  n_cols = size(Uᴿ)[2]
  level = length(Aᴿ)
  vᴿ = softmax(m.vᴿ)
  vᶜ = softmax(m.vᶜ)
  λᴿ = max(0., m.λᴿ)
  λᶜ = max(0., m.λᶜ)
  aᴿ = softmax(λᴿ * (Uᴿ * Aᶜ[1]))
  aᶜ = softmax(λᶜ * (Uᶜ * Aᴿ[1]))
  return Uᴿ, Uᶜ, [Aᴿ..., aᴿ], [Aᶜ..., aᶜ]
end

struct Last
  v
end

(m::Last)(t) = m(t...)

Last(level::Integer) = Last(randn(level))

(m::Last)(Uᴿ, Uᶜ, Aᴿ, Aᶜ) =  begin
  level = length(Aᴿ)
  v = softmax(m.v)
  return Aᴿ[1]*v[1] + Aᴿ[2]*v[2]
end

Flux.@functor Game_Dense
Flux.@functor Game_Soft
Flux.@functor Action_Response
Flux.@functor Last


Feats_data = Array{Tuple{Tuple{Array{Real,2},Array{Real,2},Array{Array{Float64,1},2}},Array{Float64,1}},1}
Features = Tuple{Array{Real,2},Array{Real,2},Array{Array{Float64,1},2}}

#%%
# pos_games, pos_play = invert(all_data[:pos])


# pos_feats = gen_feats.(pos_games);

#%%

# deep_base = Chain(Game_Dense(15,50), Game_Dense(50, 20), x -> sum(x[3][1]));
# @btime deep_base.(pos_feats);

# For some damn reason, this does not work with two Game_Dense layers. Or rather, it calculates stuff perfectly, but I can't train. Get dimension error!!!!!!
# model = Chain(Game_Dense(15,100), Game_Soft(100), Action_Response(1), Last(2));
# model = deep_base;
# model = Chain(GDense(15,50), GDense(50, 30), Game_Soft(30));
# feats_d = [(f,y) for (f,y) in zip(pos_feats, pos_play)]


# @btime model(pos_games[1])
# @btime model(feats_d[1][1])
# loss(data::Feats_data) = mean(Flux.crossentropy.(model.(data[1]), data[2]))

#
#
# loss(data::Feats_data) = sum([loss(g,y) for (g,y) in data])/length(data)
# loss(x::Features, y) = Flux.crossentropy(model(x), y)
#
# # @btime loss(feats_d)
#
# # loss(feats_d[1][1], feats_d[1][2])
# # model(feats_d[1][1])
#
#
#
# opt = ADAM(0.001, (0.9, 0.999))
# opt = RMSProp()
# ps = Flux.params(model)
# evalcb() = @show(loss(feats_d))
#
#

# @Flux.epochs 100 Flux.train!(loss, ps, feats_d, opt, cb=Flux.throttle(evalcb, 20))
# Flux.train!(loss, ps, feats_d, opt, cb=Flux.throttle(evalcb, 20))
#
#
# model
#
# end
# reshape(hcat(single_feat...), (3,3,15))
# #%% Start doing stuff
# function my_softmax(x)
#     ex = exp.(x)
#     ex ./= sum(ex)
#     ex
# end
#
# # Game_Dense_Full incorporates information about the games directly, including max,min, and mean of rows and columns.
# # mutable struct Game_Dense_full
# #   Wᴿ
# #   WPᴿ # Paramaters for incorporating game information for row player
# #   Wᶜ
# #   WPᶜ # Paramaters for incorporating game information for column player
# #   bᴿ
# #   bᶜ
# #   σ
# # end
#
# # Only looks at previous layer
#
# struct GDense{F,S,f,s,T}
#   Wᴿ::F
#   bᴿ::S
#   Wᶜ::f
#   bᶜ::s
#   σ::T
# end
#
# GDense(in::Integer, out::Integer, σ = identity) = begin
#   # Wᴿ = Flux.param(0.2 .* randn(out, in))
#   # Wᶜ = Flux.param(0.2 .* randn(out, in))
#   # bᴿ = Flux.param(0.2 .* randn(out))
#   # bᶜ = Flux.param(0.2 .* randn(out))
#   Wᴿ = 0.2 .* randn(out, in)
#   Wᶜ = 0.2 .* randn(out, in)
#   bᴿ = 0.2 .* randn(out)
#   bᶜ = 0.2 .* randn(out)
#   GDense(Wᴿ, bᴿ, Wᶜ, bᶜ, σ)
# end
#
# (m::GDense)(t) = m(t...)
# (m::GDense)(Uᴿ, Uᶜ, H) = m(Uᴿ, Uᶜ, H, H)
#
# m = GDense(15, 50)
# function apply_GDense(W, b, σ, H::AbstractArray)
#     σ.(W*H .+ b)
# end
#
#
# (m::GDense)(Uᴿ, Uᶜ, Hᴿ, Hᶜ) = begin
#     fᴿ(H::AbstractArray) = apply_GDense(m.Wᴿ, m.bᴿ, m.σ, H)
#     fᶜ(H::AbstractArray) = apply_GDense(m.Wᶜ, m.bᶜ, m.σ, H)
#     Uᴿ, Uᶜ, fᴿ.(Hᴿ), fᶜ.(Hᶜ)
# end
#
# Flux.@functor GDense
# Flux.params(m)
#
#
#
#
# m = Chain(Game_Dense(15, 50), Game_Dense(50,30), Game_Soft(30))
#
# single_feat = pos_feats[1]
# m(single_feat)
#
#
# @btime m.(pos_feats)
#
#
# # Handling some weird error where sometimes the output is sent as a tuple
#
# m1 = Game_Dense(15, 50)
# m2 = Game_Dense(50, 30)
# ms = Game_Soft(30)
#
#
# Uᴿ, Uᶜ, Hᴿ, Hᶜ = m1(pos_feats[1])
# Uᴿ, Uᶜ, Hᴿ, Hᶜ = m2(Uᴿ, Uᶜ, Hᴿ, Hᶜ)
# ms(Uᴿ, Uᶜ, Hᴿ, Hᶜ)[3][1]
#
#
# @btime m_s(Uᴿ, Uᶜ, Hᴿ, Hᶜ)[1]
#
#
# @btime val_mat = reshape(hcat(Hᶜ...), (:,3,3))
# val_mat = reshape(hcat(Hᶜ...), (:,3,3))
# S1 = reshape(sum(val_mat, dims=2), (:,3))
# Preds_R = softmax(S1;dims=2)
# S2 = reshape(sum(val_mat, dims=2), (:,3))
# softmax(S2;dims=2)
#
# [[sum([Hᶜ[i,j][k] for j in 1:3]) for i in 1:3] for k in 1:50]
# [[sum([Hᶜ[i,j][k] for i in 1:3]) for j in 1:3] for k in 1:50]
#
# [[sum([Hᴿ[i,j][k] for j in 1:3]) for i in 1:3] for k in 1:50]
# [[sum([Hᴿ[i,j][k] for i in 1:3]) for j in 1:3] for k in 1:50]
