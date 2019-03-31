using Flux
include("Heuristics.jl")
using Base


# Base.vcat(a::Tracker.TrackedReal, b::Tracker.TrackedReal) = [a,b]
function my_softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

smooth_max(r) = r' * my_softmax(8 * r)

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
  WCmaxᴿ
  WRmaxᴿ
  WCmeanᴿ
  WRmeanᴿ
  WCminᴿ
  WRminᴿ
  Wᶜ
  WUᶜ
  WCmaxᶜ
  WRmaxᶜ
  WCmeanᶜ
  WRmeanᶜ
  WCminᶜ
  WRminᶜ
  bᴿ
  bᶜ
  σ
end



tensor_mul(H, W, WU, WCmax, WRmax, WCmean, WRmean, WCmin, WRmin, U, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min, b, σ) = begin
  # mapslices(x -> W * x, H, dims=3)
  [σ.(W * H[i,j] .+ WU * U[i,j] .+ WCmax*Uᶜ_max[i,j] .+ WRmax*Uᴿ_max[i,j] .+ WCmean*Uᶜ_mean[i,j] .+ WRmean*Uᴿ_mean[i,j] .+ WCmin*Uᶜ_min[i,j] .+ WRmin*Uᴿ_min[i,j] .+ b) for i in 1:size(H)[1], j in 1:size(H)[2]]

  # [[σ.(sum([W[l,k]*H[k][i,j] + WU[l]*U[i,j] + b[l] + WCmax[l,k]*smooth_max(H[k][i,:]) + WRmax[l,k]*smooth_max(H[k][:,j]) for k in 1:length(H)])) for i in 1:3, j in 1:3] for l in 1:size(W)[1]]

end

(m::Game_Dense)((Hᴿ, Hᶜ, Uᴿ, Uᶜ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min)) = begin
  # (tensor_mul(Hᴿ, m.Wᴿ, m.WCmaxᴿ, m.WRmaxᴿ, Uᶜ, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.WUᴿ, m.WCmaxᶜ, m.WRmaxᶜ, Uᴿ, m.bᶜ,m.σ), Uᴿ, Uᶜ)
  # (tensor_mul(m, Hᴿ, Uᶜ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min), tensor_mul(m,Hᶜ, Uᴿ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min,), Uᴿ, Uᶜ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min)
  (tensor_mul(Hᴿ, m.Wᴿ, m.WUᶜ, m.WCmaxᴿ, m.WRmaxᴿ, m.WCmeanᴿ, m.WRmeanᴿ, m.WCminᴿ, m.WRminᴿ, Uᶜ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.WUᴿ, m.WCmaxᶜ, m.WRmaxᶜ, m.WCmeanᶜ, m.WRmeanᶜ, m.WCminᶜ, m.WRminᶜ, Uᴿ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min, m.bᶜ,m.σ), Uᴿ, Uᶜ, Uᴿ_max, Uᶜ_max, Uᴿ_mean, Uᶜ_mean, Uᴿ_min, Uᶜ_min)
  # (tensor_mul(Hᴿ, m.Wᴿ, m.WUᶜ, Uᶜ, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.WUᴿ, Uᴿ, m.bᶜ,m.σ), Uᴿ, Uᶜ)
end
# (m::Game_Dense)(g::Game) = m((g.row, g.col, g.row, g.col))
# (m::Game_Dense)(g::Game) = m(([g.row], [g.col], g.row, g.col))
# (m::Game_Dense)(g::Game) = m((g.row, g.col, g.row, g.col, mapslices(x -> maximum(x)*ones(length(x)), g.row, dims=1),  mapslices(x -> maximum(x)*ones(length(x)), g.col, dims=2),  mapslices(x -> mean(x)*ones(length(x)), g.row, dims=1),  mapslices(x -> mean(x)*ones(length(x)), g.col, dims=2),  mapslices(x -> minimum(x)*ones(length(x)), g.row, dims=1),  mapslices(x -> minimum(x)*ones(length(x)), g.col, dims=2)))
(m::Game_Dense)(g::Game) = m((g.row, g.col, g.row, g.col, mapslices(x -> maximum(x)*ones(length(x)), g.row, dims=2),  mapslices(x -> maximum(x)*ones(length(x)), g.col, dims=1),  mapslices(x -> mean(x)*ones(length(x)), g.row, dims=2),  mapslices(x -> mean(x)*ones(length(x)), g.col, dims=1),  mapslices(x -> minimum(x)*ones(length(x)), g.row, dims=2),  mapslices(x -> minimum(x)*ones(length(x)), g.col, dims=1)))
#
# (m::Game_Dense)(g::Game) = begin
#   Hᴿ = similar(g.row, size(g.row)...,1)
#   Hᴿ[:,:,1] = g.row
#   Hᶜ = similar(g.col, size(g.col)...,1)
#   Hᶜ[:,:,1] = g.col
#   m((Hᴿ, Hᶜ, g.row, g.col))
# end
Game_Dense(in::Integer, out::Integer, σ = identity) = begin
  Wᴿ = param(0.2 .* randn(out, in))
  Wᶜ = param(0.2 .* randn(out, in))
  WUᴿ = param(0.2 .* randn(out))
  WCmaxᴿ = param(0.2 .* randn(out))
  WRmaxᴿ = param(0.2 .* randn(out))
  WCmeanᴿ = param(0.2 .* randn(out))
  WRmeanᴿ = param(0.2 .* randn(out))
  WCminᴿ = param(0.2 .* randn(out))
  WRminᴿ = param(0.2 .* randn(out))
  # WCmaxᴿ = param(0.2 .* randn(out, in))
  # WRmaxᴿ = param(0.2 .* randn(out, in))
  WUᶜ = param(0.2 .* randn(out))
  WCmaxᶜ = param(0.2 .* randn(out))
  WRmaxᶜ = param(0.2 .* randn(out))
  WCmeanᶜ = param(0.2 .* randn(out))
  WRmeanᶜ = param(0.2 .* randn(out))
  WCminᶜ = param(0.2 .* randn(out))
  WRminᶜ = param(0.2 .* randn(out))
  # WCmaxᶜ = param(0.2 .* randn(out, in))
  # WRmaxᶜ = param(0.2 .* randn(out, in))
  bᴿ = param(0.2 .* randn(out))
  bᶜ = param(0.2 .* randn(out))
  Game_Dense(Wᴿ, WUᴿ, WCmaxᴿ,WRmaxᴿ, WCmeanᴿ,WRmeanᴿ, WCminᴿ,WRminᴿ, Wᶜ, WUᶜ, WCmaxᶜ, WRmaxᶜ, WCmeanᶜ, WRmeanᶜ, WCminᶜ, WRminᶜ, bᴿ, bᶜ, σ)
  # Game_Dense(Wᴿ, WUᴿ, Wᶜ, WUᶜ, bᴿ, bᶜ, σ)
end

mutable struct Game_Soft
  Wᴿ
  Wᶜ
end

(m::Game_Soft)((Hᴿ, Hᶜ, Uᴿ, Uᶜ)) = begin
  n_rows = size(Hᴿ)[1]
  n_cols = size(Hᴿ)[2]
  # n_rows = size(Hᴿ[1])[1]
  # n_cols = size(Hᴿ[1])[2]
  # wᴿ = max.(0,m.Wᴿ)
  # wᴿ = wᴿ/sum(wᴿ)
  wᴿ = softmax(m.Wᴿ)
  fᴿ = [my_softmax(sum([Hᴿ[i,j][k] for i in 1:n_rows, j in 1:n_cols], dims=2)) for k in 1:length(Hᴿ[1,1])]
  # fᴿ = [my_softmax(sum(Hᴿ[:,:,k], dims=2)) for k in 1:size(Hᴿ)[3]]
  # fᴿ = [my_softmax(sum(h, dims=2)) for h in Hᴿ]
  # fᴿ = mapslices(x -> my_softmax(sum(x, dims=2)), Hᴿ, dims=(1,2))
  aᴿ = [sum([fᴿ[k][i]*wᴿ[k] for k in 1:length(fᴿ)]) for i in 1:n_rows]
  # wᶜ = max.(0,m.Wᶜ)
  # wᶜ = wᶜ/sum(wᶜ)
  wᶜ = softmax(m.Wᶜ)
  fᶜ = [my_softmax(transpose(sum([Hᶜ[i,j][k] for i in 1:n_rows, j in 1:n_cols], dims=1))) for k in 1:length(Hᴿ[1,1])]
  # fᶜ = [my_softmax(transpose(sum(Hᶜ[:,:,k], dims=1))) for k in 1:size(Hᶜ)[3]]
  # fᶜ = [my_softmax(transpose(sum(h, dims=1))) for h in Hᶜ]
  aᶜ = [sum([fᶜ[k][i]*wᶜ[k] for k in 1:length(fᶜ)]) for i in 1:n_cols]
  return ([aᴿ], [aᶜ], Uᴿ, Uᶜ)
end

Game_Soft(in::Integer) = Game_Soft(param(randn(in)), param(randn(in)))


mutable struct Action_Response
  vᴿ
  vᶜ
  λᴿ
  λᶜ
end

Action_Response(level::Integer) = Action_Response(randn(level), randn(level), 3*rand(), 3*rand())
(m::Action_Response)((Aᴿ, Aᶜ, Uᴿ, Uᶜ)) =  begin
  n_rows = size(Uᴿ)[1]
  n_cols = size(Uᴿ)[2]
  level = length(Aᴿ)
  # vᴿ = max.(0., m.vᴿ)
  # vᴿ = vᴿ/sum(vᴿ)
  vᴿ = my_softmax(m.vᴿ)
  # vᶜ = max.(0., m.vᶜ)
  # vᶜ = vᶜ/sum(vᶜ)
  vᶜ = my_softmax(m.vᶜ)
  λᴿ = max(0., m.λᴿ)
  λᶜ = max(0., m.λᶜ)
  aᴿ = my_softmax([sum([m.λᴿ * sum(Uᴿ[i,:] .* Aᶜ[l]) * vᴿ[l] for l in 1:level]) for i in 1:n_rows])
  aᶜ = my_softmax([sum([m.λᶜ *  sum(Aᴿ[l] .* Uᶜ[:,i]) * vᶜ[l] for l in 1:level]) for i in 1:n_rows])
  return ([Aᴿ..., aᴿ], [Aᶜ..., aᶜ], Uᴿ, Uᶜ)
end

mutable struct Last
  v
end

Last(level::Integer) = Last(randn(level))
(m::Last)((Aᴿ, Aᶜ, Uᴿ, Uᶜ)) =  begin
  level = length(Aᴿ)
  # v = max.(0., m.v)
  # v = v/sum(v)
  v = my_softmax(m.v)
  return sum(Aᴿ[l] * v[l] for l in 1:level)
end


Flux.@treelike Game_Dense
Flux.@treelike Game_Soft
Flux.@treelike Action_Response
Flux.@treelike Last


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
