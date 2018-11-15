include("PHeuristics.jl")
using Optim

struct Costs
    α::Float64
    λ::Float64
end

function cost(h::Heuristic, c::Costs)
    cost = abs(h.λ) * c.λ
    cost += 2(sigmoid(abs(5h.α)) - 0.5) * c.α
    cost
end
sigmoid(x) = (1. ./ (1. .+ exp.(-x)))


SimHeuristic(x::Vector{T} where T <: Real) = begin
     map(1:3:length(x)) do i
        Heuristic(x[i:i+2]...)
    end |> SimHeuristic
end

function loss(h::SimHeuristic, games, opp_plays, costs::Costs)
    pay = 0
    for i in eachindex(games)
        p = decide_probs(h, games[i])
        pay += p' * opp_plays[i]
    end
    -(pay / length(games) - sum(cost(h, costs) for h in h.h_list))
end
function loss(x::Vector{T} where T <: Real, games, opp_plays, costs)
    loss(SimHeuristic(x), games, opp_plays, costs)
end

function optimize_h(level, games, opp_plays, costs; init_x=nothing)
    loss_wrap(x) = loss(x, games, opp_plays, costs)
    if init_x == nothing
        init_x = ones(3 * level) * 0.1
    end
    x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
    SimHeuristic(x)
end
function optimize_h(level, ρ, game_size, opp_h, costs; n_games=1000)
    games = [Game(game_size, ρ) for i in range(1,n_games)]
    opp_plays = opp_cols_played(opp_h, games);
    optimize_h(level, games, opp_plays, costs)
end


ρ = 0.5
game_size = 3
n_games = 1000
games = [Game(game_size, ρ) for i in range(1,n_games)]
opp_h = Heuristic(0, 0, 10)
opp_plays = opp_cols_played(opp_h, games)
costs = Costs(0.01, 0.01)

init = [0, 0.1, 1.79, 0, 0.55, 1.81]
init = ones(6) * .1
h2 = optimize_h(2, games, opp_plays, costs; init_x=init)
println(h2)
println(loss(h2, games, opp_plays, costs))


h3 = optimize_h(3, games, opp_plays, costs)
println(h2)
println(loss(h2, games, opp_plays, costs))
println(h3)
println(loss(h3, games, opp_plays, costs))

h2 = optimize_h(2, 0.5, 3, Heuristic(0, 0, 10), Costs(0.01, 0.01))
h3 = optimize_h(3, 0.5, 3, Heuristic(0, 0, 10), Costs(0.01, 0.01))


g = Game(3, 0.5)
h1 = deepcopy(h)
h1.h_list[2].λ = 1
decide_probs(h1, g)
h1.level

@time Optim.minimizer(optimize(loss, ones(3) * 0.1, BFGS(); autodiff = :forward))
@time Optim.minimizer(optimize(loss2, ones(3) * 0.1, BFGS(); autodiff = :forward))





plot(α, sigmoid(α))
1. .+ exp.(-α)

α = 0:0.1:2
γ = -3:0.1:3
X = [loss([α, γ, 3.]) for α=α, γ=γ]

heatmap(-X, xlabel="gamma", ylabel="alpha")
idx = 1:5:length(α)
plot!(yticks=(idx, α[idx]))
idx = 1:5:length(γ)
plot!(xticks=(idx, γ[idx]))
    # xticks = (1:5:101, -4:1:4),
    # yticks = (1:5:101, -3:1:3)
opp_h
size(X)

xticks
X *= -1
X /= n_games

using Plots
heatmap(X)

g = x -> ForwardDiff.gradient(loss, x) # g = ∇f

function sgd()
    x = [0., 0., 0.1]
    g = x -> ForwardDiff.gradient(loss, x) # g = ∇f
    for i in 1:10000
        (i % 100 == 0) && println(x, "   ", loss(x))
        x -= 0.1 * g(x)
    end
    return x
end



h = Heuristic(x...)
decide_probs(h, games[1])[2]
