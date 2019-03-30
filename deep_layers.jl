module DeepLayers
using Flux
include("Heuristics.jl")
using Base

export Game_Dense, Game_Soft, Affine

# Base.vcat(a::Tracker.TrackedReal, b::Tracker.TrackedReal) = [a,b]
function my_softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

mutable struct Affine
  W
  b
  σ
end

Flux.@treelike Affine

Affine(in::Integer, out::Integer, σ = identity) = Affine(param(randn(out, in)), param(randn(out)), σ)

(m::Affine)(x) = m.σ.(m.W * x .+ m.b)


mutable struct Game_Dense
  Wᴿ
  WUᴿ
  Wᶜ
  WUᶜ
  bᴿ
  bᶜ
  σ
end

tensor_mul(H, W, WU, U, b, σ) = [σ.(W * H[i,j] .+ WU.*U[i,j] .+ b) for i in 1:size(H)[1], j in 1:size(H)[2]]

(m::Game_Dense)((Hᴿ, Hᶜ, Uᴿ, Uᶜ)) = begin
  (tensor_mul(Hᴿ, m.Wᴿ, m.WUᶜ, Uᶜ, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.WUʳ, Uᴿ, m.bᶜ,m.σ), Uᴿ, Uᶜ)
end

(m::Game_Dense)(g::Game) = m((g.row, g.col, g.row, g.col))

Game_Dense(in::Integer, out::Integer, σ = identity) = begin
  Wᴿ = param(randn(out, in))
  Wᶜ = param(randn(out, in))
  WUᴿ = param(randn(out))
  WUᶜ = param(randn(out))
  bᴿ = param(randn(out))
  bᶜ = param(randn(out))
  Game_Dense(Wᴿ, WUᴿ, Wᶜ, WUᶜ, bᴿ, bᶜ, σ)
end

mutable struct Game_Soft
  Wᴿ
  Wᶜ
end

(m::Game_Soft)((Hᴿ, Hᶜ, Uᴿ, Uᶜ)) = begin
  n_row = size(Hᴿ)[1]
  n_col = size(Hᴿ)[2]
  wᴿ = max.(0,m.Wᴿ)
  wᴿ = wᴿ/sum(wᴿ)
  fᴿ = [my_softmax(sum([Hᴿ[i,j][k] for i in 1:n_row, j in 1:n_col], dims=2)) for k in 1:length(Hᴿ[1,1])]
  aᴿ = [sum([fᴿ[k][i]*wᴿ[k] for k in 1:length(fᴿ)]) for i in 1:n_row]

  wᶜ = max.(0,m.Wᶜ)
  wᶜ = wᶜ/sum(wᶜ)
  fᶜ = [my_softmax(transpose(sum([Hᶜ[i,j][k] for i in 1:n_row, j in 1:n_col], dims=1))) for k in 1:length(Hᶜ[1,1])]
  aᶜ = [sum([fᶜ[k][i]*wᶜ[k] for k in 1:length(fᶜ)]) for i in 1:n_col]

  return (aᴿ, aᶜ, Uᴿ, Uᶜ)
end

Game_Soft(in::Integer) = Game_Soft(param(randn(1,in)), param(randn(1, in)))


Flux.@treelike Game_Dense
Flux.@treelike Game_Soft


#
# m = Chain(Game_Dense(1,10, sigmoid), Game_Dense(10,4), Game_Soft(4))
# m = Chain(Game_Dense(1,10, sigmoid), Game_Dense(10,4))
# g = Game(3, 0.3)
# m(g)[1]
# # m = Chain(Game_Dense(1,10, sigmoid), Game_Dense(10,4))
# g = Game(3, 0.3)
# msoft = Game_Soft(4)
#
# Hᴿ = m(g)[1]
# Hᶜ = m(g)[2]
#
# n_row = size(Hᴿ)[1]
# n_col = size(Hᴿ)[2]
# k = 2
# Tracker.collect(my_softmax(vec(sum([Hᴿ[i,j][k] for i in 1:n_row, j in 1:n_col], dims=2))))
# fᴿ = Tracker.collect([Tracker.collect(my_softmax(sum([Hᴿ[i,j][k] for i in 1:n_row, j in 1:n_col], dims=2))) for k in 1:length(Hᴿ[1,1])])
# msoft.Wᴿ * fᴿ
# aᴿ = msoft.Wᴿ * fᴿ/sum(msoft.Wᴿ)
# aᴿ = max.(0.01, aᴿ[1])
# aᴿ = aᴿ/sum(aᴿ)
#
# fᶜ = [my_softmax(transpose(sum([Hᶜ[i,j][k] for i in 1:n_row, j in 1:n_col], dims=1))) for k in length(Hᶜ[1,1])]
# aᶜ = m.Wᶜ * fᶜ/sum(m.Wᶜ)
#
#
# m = [randn(10) for i in 1:3, j in 1:3]
# w = randn(10,10)
# tensor_mul(m, w, 1)
#
# vcat(m[1,2],1)
# m[1,2,:] = w * m[1,2,:]
#
# sigmoid.(m)
#
# f((x,y)) = x+2, y+2
# f(f((1,1)))
#
# g = Game(3,0.1)
# g.col
# g.row
#
# [m[1,j][2] for j in 1:3]
#
# m = randn(3,3,10)
# m[1,2,:]
#
# m = [randn(10) for i in 1:3, j in 1:3]
# a = softmax(sum([m[i,j][2] for i in 1:3, j in 1:3], dims=2))
# sum(a,dims=2)
# a
# Hᴿ = m
# k = 3
#
# my_softmax(sum([Hᴿ[i,j][k] for i in 1:3, j in 1:3], dims=2))
# f_vec = [softmax(transpose(sum([Hᴿ[i,j][k] for i in 1:3, j in 1:3], dims=1))) for k in 1:length(Hᴿ[1,1])]
# sum([Hᴿ[i,j][1] for i in 1:3, j in 1:3], dims=2)
# w = rand(1, 10)
# w * f_vec/sum(w)
