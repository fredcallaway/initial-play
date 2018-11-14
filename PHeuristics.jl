import Distributions: MvNormal
import Distributions
import StatsFuns: softmax
import StatsBase: sample, Weights
import Base
import DataStructures: OrderedDict
using BlackBoxOptim
import Printf: @printf, @sprintf

function sample_cell(ρ; max=9.5, min=-0.5, µ=5, σ=5)
    if ρ < 1 && ρ > -1
        Σ = [[1 ρ]
             [ρ 1]]
        dist = MvNormal(Σ)
        r, c = rand(dist)*σ .+ μ
        while any([r, c] .< min) || any([r, c] .> max)
             r, c = rand(dist)*σ .+ μ
        end
    else
        r = randn()*σ + μ
        c = ρ*(r - μ) + μ
        while any([r, c] .< min) || any([r, c] .> max)
            r = randn()*σ + μ
            c = ρ*(r - μ) + μ
        end
    end
    r = round(r)
    c = round(c)
    return (r, c)
end

struct Game
    row::Matrix{Float64}
    col::Matrix{Float64}
end

Game(size::Int, ρ::Number) = begin
    X = [sample_cell(ρ) for i in 1:size, j in 1:size]
    Game(
        map(x->x[1], X),
        map(x->x[2], X),
    )
end

Base.transpose(g::Game) = Game(transpose(g.col), transpose(g.row))
Base.size(g::Game) = size(g.row)[1]

Base.show(io::IO, g::Game) = begin
    for i in 1:size(g)
        for j in 1:size(g)
            print(Int(g.row[i,j]), ",", Int(g.col[i,j]), "  ")
        end
        println()
    end
end


function weighted_geo_mean(a,b, α)
    return exp(α*log(a + 0.001) + (1-α)*log(b + 0.001))
end

function p_norm(vec::Vector{Float64}, p::Float64)
    val = sum(vec.^p)
    return val^(1/p)
end

mutable struct Heuristic
    α::Float64
    γ::Float64
    λ::Float64
end

mutable struct SimHeuristic
    h_list::Vector{Heuristic}
    level::Int64
end



Heuristic(dists::OrderedDict) = Heuristic(map(rand, dists.vals)...)

SimHeuristic(dists::OrderedDict, level::Int64) = SimHeuristic([Heuristic(map(rand, dists.vals)...) for i in 1:level], level)

Base.show(io::IO, h::Heuristic) = @printf(io, "Heuristic: α=%.2f, γ=%.2f, λ=%.2f", h.α, h.γ, h.λ)

h_dists = OrderedDict(
    "α" => Distributions.Uniform(0,1),
    "γ" => Distributions.Uniform(-10,10),
    "λ" => Distributions.Uniform(0,10),
)

function relative_values(h::Heuristic, game::Game)
    map(1:size(game)) do i
        r = game.row[i, :]
        c = game.col[i, :]
        s = map((r, c) -> r / (1 + exp(-h.α * c)), r, c)
        v = s' * softmax(h.γ * s)
    end
end

function decide(h::Heuristic, game::Game)
    v = relative_values(h, game)
    v = softmax(h.λ * v)
    choice = sample(1:length(v), Weights(v))
end

function decide_probs(h::Heuristic, game::Game)
    v = relative_values(h, game)
    v = softmax(h.λ*v)
end


function decide(s::SimHeuristic, game::Game)
    self_g = deepcopy(game)
    opp_g = deepcopy(transpose(game))
    choice = 0
    for i in 1:s.level
        if i == s.level
            choice = decide(s.h_list[i], self_g)
        else
            opp_pred = decide_probs(s.h_list[i], opp_g)
            self_pred = decide_probs(s.h_list[i], self_g)
            self_g.row .*= opp_pred'
            opp_g.row .*= self_pred'
        end
    end
    # println("Done, i=%f", i)
    return choice
end



function payoff(h, opp_h, games)
    payoff = 0
    for g in games
        decision = decide(h, g)
        opp_decision = decide(opp_h, transpose(g))
        payoff += g.row[decision, opp_decision]
    end
    return payoff
end


opp_cols_played = (opp_h, games) -> [g.row[:,decide(opp_h, transpose(g))] for g in games]

function fitness_from_col_play(h, games, opp_plays)
    payoff = 0
    for i in 1:length(games)
        decision = decide(h, games[i])
        payoff += opp_plays[i][decision]
    end
    return payoff
end


function opt_h!(h::Heuristic, opp_h::Heuristic, games, h_dists; max_time=20.0)
    opp_plays = opp_cols_played(opp_h, games)
    param_names = [:α, :γ, :λ]
    init_params = [h.α, h.γ, h.λ]
    function opt_fun(params)
        for (name, param) in zip(param_names, params)
            setfield!(h, name, param)
        end
        return -fitness_from_col_play(h, games, opp_plays)
    end
    # res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 4, MaxTime=10.0, Method = :separable_nes, TraceMode = :silent, TraceInterval =5.0)
    res = bboptimize(opt_fun; SearchRange = [(h_dists["α"].a ,h_dists["α"].b), (h_dists["γ"].a ,h_dists["γ"].b), (h_dists["λ"].a ,h_dists["λ"].b)], MaxTime=max_time, TraceMode = :silent, TraceInterval =4.0)
    for (name, param) in zip(param_names, best_candidate(res))
        setfield!(h, name, param)
    end
end

function opt_s!(s::SimHeuristic, opp_h::Heuristic, games, h_dists; max_time=20.0)
    opp_plays = opp_cols_played(opp_h, games)
    n_h_params = length(fieldnames(Heuristic))
    param_names = repeat([:α, :γ, :λ], outer=[s.level])
    h = s.h_list[1]
    init_params = repeat([h.α, h.γ, h.λ], outer=[s.level])
    function opt_fun(params)
        for i in 1:s.level
            for j in 1:n_h_params
                param_idx = (i-1)*n_h_params + j
                setfield!(s.h_list[i], param_names[j], params[param_idx])
            end
        end
        return -fitness_from_col_play(s, games, opp_plays)
    end
    res = bboptimize(opt_fun; SearchRange = repeat([(h_dists["α"].a ,h_dists["α"].b), (h_dists["γ"].a ,h_dists["γ"].b), (h_dists["λ"].a ,h_dists["λ"].b)], outer=[s.level]), MaxTime=max_time, TraceMode = :silent, TraceInterval =4.0)
    res_params = best_candidate(res)
    for i in 1:s.level
        for j in 1:n_h_params
            param_idx = (i-1)*n_h_params + j
            setfield!(s.h_list[i], param_names[j], res_params[param_idx])
        end
    end
end

h = Heuristic(0, 0, 0.0)
g = Game(3, 0.)
g.row[1,2] = 100
g
relative_values(h, g)

s = SimHeuristic(h_dists, 2)
s3 = SimHeuristic(h_dists, 3)

prisoners_dilemma = Game([[3 0];[4 1]], [[3 4]; [0 1]])


level_0 = Heuristic(0.5, 1., 0.0)
level_1 = Heuristic(1., 1., 4)
noisy = Heuristic(0.5, 2., 0.9)
maximin = Heuristic(1., -10., 10.)
maxjoint = Heuristic(0.5, 10., 10.)


ρ = -0.99
n_games = 1000
game_size = 3
opp_h = level_1
println("Compare with ρ = ", ρ)
training_games = [Game(game_size, ρ) for i in range(1,n_games)];


test_games = [Game(game_size, ρ) for i in range(1,n_games)];
println("Level-0: \t", payoff(level_0, opp_h, training_games))
println("Level-1: \t", payoff(level_1, opp_h, training_games))
println("Maximin: \t", payoff(maximin, opp_h, training_games))
println("Max joint: \t",payoff(maxjoint, opp_h, training_games))

opt_h!(h, opp_h, training_games, h_dists; max_time=10.)
println(h, "    ", payoff(h, opp_h, training_games))

function print_result(h)
    println("-"^70)
    println(h)
    println("train score ", payoff(h, opp_h, training_games))
    println("test score  ", payoff(h, opp_h, test_games))
end
for i in 1:3
    training_games = [Game(game_size, ρ) for i in range(1,n_games)];
    test_games = [Game(game_size, ρ) for i in range(1,n_games)];
    opt_h!(h, opp_h, training_games, h_dists; max_time=10.)
    println("-"^70)
    println(h)
    println("train score ", payoff(h, opp_h, training_games))
    println("test score  ", payoff(h, opp_h, test_games))
end


opt_s!(s, opp_h, training_games, h_dists; max_time=20.)
println(s, "    ", payoff(s, opp_h, training_games))
opt_s!(s3, opp_h, training_games, h_dists; max_time=30.)
println(s3,"    ",  payoff(s3, opp_h, training_games))

g = Game(3, 0.5)

for x in (0.1:.1:1)
    p_norm(Float64[1, 5, 10], x)
end
[h, h]
println("Test games:")

println("Best h: \t", payoff(h, opp_h, test_games))
println("Best s2: \t", payoff(s, opp_h, test_games))
println("Best s3: \t", payoff(s3, opp_h, test_games))
println("Level-0: \t", payoff(level_0, opp_h, test_games))
println("Level-1: \t", payoff(level_1, opp_h, test_games))
# println("Level-2: \t", payoff(level_2, opp_h, test_games))
println("Maximin: \t", payoff(maximin, opp_h, test_games))
println("Sum joint: \t",payoff(maxjoint, opp_h, test_games))


println(@sprintf("PD for h: %0.f", decide(h, prisoners_dilemma)))
# level_2 = SimHeuristic(level_1, level_1, 4.)
# maximin = Heuristic(1,1,0,0,4, (+), minimum)
# sum_prod = Heuristic(1,1,1,1,4, (*), sum)
# max_prod = Heuristic(1,1,1,1,4, (*), maximum)
# max_sum = Heuristic(1,1,1,1,4, (+), maximum)
# sum_sum = Heuristic(1,1,1,1,4, (+), sum)
