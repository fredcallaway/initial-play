using Flux
include("Heuristics.jl")
using Base


function my_softmax(x)
    ex = exp.(x)
    ex ./= sum(ex)
    ex
end

# Game_Dense_Full incorporates information about the games directly, including max,min, and mean of rows and columns.
mutable struct Game_Dense_full
  Wᴿ
  WPᴿ # Paramaters for incorporating game information for row player
  Wᶜ
  WPᶜ # Paramaters for incorporating game information for column player
  bᴿ
  bᶜ
  σ
end

# Only looks at previous layer
mutable struct Game_Dense
  Wᴿ
  Wᶜ
  bᴿ
  bᶜ
  σ
end

# Update functions if game information is directly incorporated
tensor_mul(H, W, WP, US, b, σ) = begin
  [σ.(W * H[i,j] .+ sum([w *u[i,j] for (w,u) in zip(WP,US)]) .+ b) for i in 1:size(H)[1], j in 1:size(H)[2]]
end

# Update function with only previous layer
tensor_mul(H, W, b, σ) = begin
  [σ.(W * H[i,j] .+ b) for i in 1:size(H)[1], j in 1:size(H)[2]]
end

# Handling some weird error where sometimes the output is sent as a tuple
(m::Game_Dense)(t) = m(t...)
(m::Game_Dense_full)(t) = m(t...)

(m::Game_Dense_full)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = begin
  tensor_mul(Hᴿ, m.Wᴿ, m.WPᴿ, US, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.WPᶜ, US, m.bᶜ,m.σ), US, Uᴿ, Uᶜ
end

(m::Game_Dense)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = begin
  tensor_mul(Hᴿ, m.Wᴿ, m.bᴿ, m.σ), tensor_mul(Hᶜ, m.Wᶜ, m.bᶜ,m.σ), US, Uᴿ, Uᶜ
end

# Allows us to take games directly as input. Also generates a vector with different transformations of payoff matrices
# such as max of rows/columns.
(m::Game_Dense)(g::Game) = m(g.row, g.col, [g.row, g.col, [min(g.row[i,j], g.col[i,j]) for i in 1:3, j in 1:3], [mapslices(x -> f(x)*ones(length(x)), u, dims=d) for f in [maximum, mean, minimum], d in [1,2], u in [g.row, g.col]]...], g.row, g.col)
(m::Game_Dense_full)(g::Game) = m(g.row, g.col, [g.row, g.col, [min(g.row[i,j], g.col[i,j]) for i in 1:3, j in 1:3], [mapslices(x -> f(x)*ones(length(x)), u, dims=d) for f in [maximum, mean, minimum], d in [1,2], u in [g.row, g.col]]...], g.row, g.col)

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
  WPᴿ = [param(randn(out)) for i in 1:15]
  WPᶜ = [param(randn(out)) for i in 1:15]
  bᴿ = param(0.2 .* randn(out))
  bᶜ = param(0.2 .* randn(out))
  Game_Dense_full(Wᴿ, WPᴿ, Wᶜ, WPᶜ, bᴿ, bᶜ, σ)
end

# Turns a previous layer of matrices into a a single prediction based on a weighted softmax of each previous layer
mutable struct Game_Soft
  Wᴿ
  Wᶜ
end

(m::Game_Soft)(t) = m(t...)
(m::Game_Soft)(Hᴿ, Hᶜ, US, Uᴿ, Uᶜ) = begin
  n_rows = size(Hᴿ)[1]
  n_cols = size(Hᴿ)[2]
  wᴿ = softmax(m.Wᴿ) # To force it into the simplex
  fᴿ = [my_softmax(sum([Hᴿ[i,j][k] for i in 1:n_rows, j in 1:n_cols], dims=2)) for k in 1:length(Hᴿ[1,1])]
  aᴿ = [sum([fᴿ[k][i]*wᴿ[k] for k in 1:length(fᴿ)]) for i in 1:n_rows]

  wᶜ = softmax(m.Wᶜ)
  fᶜ = [my_softmax(transpose(sum([Hᶜ[i,j][k] for i in 1:n_rows, j in 1:n_cols], dims=1))) for k in 1:length(Hᴿ[1,1])]
  aᶜ = [sum([fᶜ[k][i]*wᶜ[k] for k in 1:length(fᶜ)]) for i in 1:n_cols]
  return [aᴿ], [aᶜ], Uᴿ, Uᶜ
end

Game_Soft(in::Integer) = Game_Soft(param(randn(in)), param(randn(in)))

# Action response layers weights the prediction of previous levels and logit-best responds
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
  vᴿ = my_softmax(m.vᴿ)
  vᶜ = my_softmax(m.vᶜ)
  λᴿ = max(0., m.λᴿ)
  λᶜ = max(0., m.λᶜ)
  aᴿ = my_softmax([sum([λᴿ * sum(Uᴿ[i,:] .* Aᶜ[l]) * vᴿ[l] for l in 1:level]) for i in 1:n_rows])
  aᶜ = my_softmax([sum([λᶜ * sum(Aᴿ[l] .* Uᶜ[:,i]) * vᶜ[l] for l in 1:level]) for i in 1:n_cols])
  return [Aᴿ..., aᴿ], [Aᶜ..., aᶜ], Uᴿ, Uᶜ
end

# The last layer weights all actions response layers, including the "0" layer from Game_Soft
mutable struct Last
  v
end

(m::Last)(t) = m(t...)

Last(level::Integer) = Last(randn(level))

(m::Last)(Aᴿ, Aᶜ, Uᴿ, Uᶜ) =  begin
  level = length(Aᴿ)
  v = my_softmax(m.v)
  return sum(Aᴿ[l] * v[l] for l in 1:level)
end


# Makes the custom layers accesible for flux functions such as params, or Chain
Flux.@treelike Game_Dense
Flux.@treelike Game_Dense_full
Flux.@treelike Game_Soft
Flux.@treelike Action_Response
Flux.@treelike Last
