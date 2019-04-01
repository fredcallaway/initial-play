using Flux
include("Heuristics.jl")
using Base


function my_softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end


mutable struct Game_Dense_full
  Wᴿ
  WPᴿ
  Wᶜ
  WPᶜ
  bᴿ
  bᶜ
  σ
end

mutable struct Game_Dense
  Wᴿ
  Wᶜ
  bᴿ
  bᶜ
  σ
end

tensor_mul(H, W, WP, US, b, σ) = begin
  [σ.(W * H[i,j] .+ sum([w *u[i,j] for (w,u) in zip(WP,US)]) .+ b) for i in 1:size(H)[1], j in 1:size(H)[2]]
end

tensor_mul(H, W, b, σ) = begin
  [σ.(W * H[i,j] .+ b) for i in 1:size(H)[1], j in 1:size(H)[2]]
end


(m::Game_Dense)(t) = m(t...)
(m::Game_Dense_full)(t) = m(t...)

(m::Game_Dense_full)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = begin
  tensor_mul(Hᴿ, m.Wᴿ, m.WPᴿ, US, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.WPᶜ, US, m.bᶜ,m.σ), US, Uᴿ, Uᶜ
end

(m::Game_Dense)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = begin
  tensor_mul(Hᴿ, m.Wᴿ, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.bᶜ,m.σ), US, Uᴿ, Uᶜ
end

(m::Game_Dense)(g::Game) = m(g.row, g.col, [g.row, g.col, [mapslices(x -> f(x)*ones(length(x)), u, dims=d) for f in [maximum, mean, minimum], d in [1,2], u in [g.row, g.col]]...], g.row, g.col)
(m::Game_Dense_full)(g::Game) = m(g.row, g.col, [g.row, g.col, [mapslices(x -> f(x)*ones(length(x)), u, dims=d) for f in [maximum, mean, minimum], d in [1,2], u in [g.row, g.col]]...], g.row, g.col)

Game_Dense(in::Integer, out::Integer, σ = identity) = begin
  Wᴿ = param(0.2 .* randn(out, in))
  Wᶜ = param(0.2 .* randn(out, in))
  bᴿ = param(0.2 .* randn(out))
  bᶜ = param(0.2 .* randn(out))
  Game_Dense(Wᴿ, Wᶜ, bᴿ, bᶜ, σ)
end

Game_Dense_full(in::Integer, out::Integer, σ = identity) = begin
  Wᴿ = param(0.2 .* randn(out, in))
  Wᶜ = param(0.2 .* randn(out, in))
  WPᴿ = [param(randn(out)) for i in 1:14]
  WPᶜ = [param(randn(out)) for i in 1:14]
  bᴿ = param(0.2 .* randn(out))
  bᶜ = param(0.2 .* randn(out))
  Game_Dense_full(Wᴿ, WPᴿ, Wᶜ, WPᶜ, bᴿ, bᶜ, σ)
end

mutable struct Game_Soft
  Wᴿ
  Wᶜ
end

# (m::Game_Soft)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = m(Hᴿ, Hᶜ, Uᴿ, Uᶜ)
(m::Game_Soft)(t) = m(t...)
(m::Game_Soft)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = begin
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
  return [aᴿ], [aᶜ], Uᴿ, Uᶜ
end

Game_Soft(in::Integer) = Game_Soft(param(randn(in)), param(randn(in)))


mutable struct Action_Response
  vᴿ
  vᶜ
  λᴿ
  λᶜ
end

(m::Action_Response)(t) = m(t...)
Action_Response(level::Integer) = Action_Response(randn(level), randn(level), 3*rand(), 3*rand())
(m::Action_Response)(Aᴿ, Aᶜ, Uᴿ, Uᶜ) =  begin
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
  return [Aᴿ..., aᴿ], [Aᶜ..., aᶜ], Uᴿ, Uᶜ
end

mutable struct Last
  v
end

(m::Last)(t) = m(t...)

Last(level::Integer) = Last(randn(level))

(m::Last)(Aᴿ, Aᶜ, Uᴿ, Uᶜ) =  begin
  level = length(Aᴿ)
  # v = max.(0., m.v)
  # v = v/sum(v)
  v = my_softmax(m.v)
  return sum(Aᴿ[l] * v[l] for l in 1:level)
end


Flux.@treelike Game_Dense
Flux.@treelike Game_Dense_full
Flux.@treelike Game_Soft
Flux.@treelike Action_Response
Flux.@treelike Last
