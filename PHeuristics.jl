import Distributions: MvNormal
import Distributions
import StatsFuns: softmax
import StatsBase: sample, Weights
import Statistics: mean
import Base
import DataStructures: OrderedDict
using BlackBoxOptim
import Optim
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
    "α" => Distributions.Uniform(0,10),
    "γ" => Distributions.Uniform(-10,10),
    "λ" => Distributions.Uniform(0,10),
)

function μ(mat::Matrix{Float64}, row::Int64=0)
    if row == 0
        return mean(mat)
    else
        return (mean(mat[row,:]))
    end
end

function relative_values(h::Heuristic, game::Game)
    map(1:size(game)) do i
        μ_r = μ(game.row)
        μ_c = μ(game.col)
        r = game.row[i, :] .- μ_r
        c = game.col[i, :] .- μ_c
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
        elseif (s.level - 1) % 2 == 1
            opp_pred = decide_probs(s.h_list[i], opp_g)
            self_g.row .*= opp_pred'
        elseif (s.level - 1) % 2 == 0
            self_pred = decide_probs(s.h_list[i], self_g)
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


function opt_h!(h::Heuristic, opp_h::Heuristic, games, h_dists; max_time=20.0, method=:de_rand_1_bin, trace=:silent)
    opp_plays = opp_cols_played(opp_h, games)
    param_names = [:α, :γ, :λ]
    init_params = [h.α, h.γ, h.λ]
    function opt_fun(params)
        for (name, param) in zip(param_names, params)
            setfield!(h, name, param)
        end
        return -fitness_from_col_play(h, games, opp_plays)
    end
    res = bboptimize(opt_fun; SearchRange = [(h_dists["α"].a ,h_dists["α"].b), (h_dists["γ"].a ,h_dists["γ"].b), (h_dists["λ"].a ,h_dists["λ"].b)], MaxTime=max_time, Method=method, TraceMode = trace, TraceInterval =4.0)
    for (name, param) in zip(param_names, best_candidate(res))
        setfield!(h, name, param)
    end
end

function opt_s!(s::SimHeuristic, opp_h::Heuristic, games, h_dists; max_time=20.0, method=method)
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
    res = bboptimize(opt_fun; SearchRange = repeat([(h_dists["α"].a ,h_dists["α"].b), (h_dists["γ"].a ,h_dists["γ"].b), (h_dists["λ"].a ,h_dists["λ"].b)], outer=[s.level]), MaxTime=max_time,  Method=method, TraceMode = :silent, TraceInterval =4.0)
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
prisoners_dilemma = Game(prisoners_dilemma.row .*2, prisoners_dilemma.col .*2)


level_0 = Heuristic(0.5, 1., 0.0)
level_1 = Heuristic(0., 0., 10.)
noisy = Heuristic(0.2, 1., 0.9)
maximin = Heuristic(0., -10., 10.)
maxjoint = Heuristic(0.5, 10., 10.)


ρ = 0.6
n_games = 1000
game_size = 3
method = :probabilistic_descent
opp_h = noisy
correct_level_2 = SimHeuristic([opp_h, Heuristic(0., 0., 10.)], 2)
correct_level_3 = SimHeuristic([Heuristic(0., 0., 0.), opp_h, Heuristic(0., 0., 10.)], 3)
println("Compare with ρ = ", ρ)
training_games = [Game(game_size, ρ) for i in range(1,n_games)];


test_games = [Game(game_size, ρ) for i in range(1,n_games)];
println("Level-0: \t", payoff(level_0, opp_h, training_games))
println("Level-1: \t", payoff(level_1, opp_h, training_games))
println("Level-2: \t", payoff(correct_level_2, opp_h, training_games))
println("Level-3: \t", payoff(correct_level_3, opp_h, training_games))
println("Maximin: \t", payoff(maximin, opp_h, training_games))
println("Max joint: \t",payoff(maxjoint, opp_h, training_games))

opt_h!(h, opp_h, training_games, h_dists; max_time=10., method=method)
println(h, "    ", payoff(h, opp_h, training_games))
opt_s!(s, opp_h, training_games, h_dists; max_time=20., method=method)
println(s, "    ", payoff(s, opp_h, training_games))
opt_s!(s3, opp_h, training_games, h_dists; max_time=30., method=method)
println(s3,"    ",  payoff(s3, opp_h, training_games))


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
println(@sprintf("PD for s2: %0.f", decide(s, prisoners_dilemma)))
println(@sprintf("PD for s3: %0.f", decide(s3, prisoners_dilemma)))

for ρ in -1.:0.2:1.
    println("----- ρ=", ρ, " -----")
    h = Heuristic(h_dists)
    games = [Game(game_size, ρ) for i in range(1,n_games)]
    opt_h!(h, opp_h, games, h_dists; max_time=10., method=method)
    println(h, "    ", payoff(h, opp_h, games))
    println(@sprintf("PD for h: %0.f", decide(h, prisoners_dilemma)))
    println("Level-1: \t", payoff(level_1, opp_h, games))
end

ρ = -0.5
games = [Game(game_size, ρ) for i in range(1,n_games)]
for method in [:separable_nes, :de_rand_1_bin, :de_rand_2_bin, :generating_set_search, :probabilistic_descent, :resampling_memetic_search, :generating_set_search]
    h = Heuristic(h_dists)
    opt_h!(h, opp_h, games, h_dists; max_time=10., method=method, trace=:silent)
    println("Using optimizer: ", method)
    println(h, "    ", payoff(h, opp_h, games))
end


#
# function print_result(h)
#     println("-"^70)
#     println(h)
#     println("train score ", payoff(h, opp_h, training_games))
#     println("test score  ", payoff(h, opp_h, test_games))
# end
#
# for i in 1:3
#     training_games = [Game(game_size, ρ) for i in range(1,n_games)];
#     test_games = [Game(game_size, ρ) for i in range(1,n_games)];
#     opt_h!(h, opp_h, training_games, h_dists; max_time=10.)
#     println("-"^70)
#     println(h)
#     println("train score ", payoff(h, opp_h, training_games))
#     println("test score  ", payoff(h, opp_h, test_games))
# end
#
# function print_result(h)
#     println("-"^70)
#     println(h)
#     println("train score ", payoff(h, opp_h, training_games))
#     println("test score  ", payoff(h, opp_h, test_games))
# end
#
# for i in 1:3
#     training_games = [Game(game_size, ρ) for i in range(1,n_games)];
#     test_games = [Game(game_size, ρ) for i in range(1,n_games)];
#     opt_h!(h, opp_h, training_games, h_dists; max_time=10.)
#     println("-"^70)
#     println(h)
#     println("train score ", payoff(h, opp_h, training_games))
#     println("test score  ", payoff(h, opp_h, test_games))
# end
#
#
# opt_s!(s, opp_h, training_games, h_dists; max_time=20.)
# println(s, "    ", payoff(s, opp_h, training_games))
# opt_s!(s3, opp_h, training_games, h_dists; max_time=30.)
# println(s3,"    ",  payoff(s3, opp_h, training_games))
#
#
# println("Test games:")
#
# println("Best h: \t", payoff(h, opp_h, test_games))
# println("Best s2: \t", payoff(s, opp_h, test_games))
# println("Best s3: \t", payoff(s3, opp_h, test_games))
# println("Level-0: \t", payoff(level_0, opp_h, test_games))
# println("Level-1: \t", payoff(level_1, opp_h, test_games))
# # println("Level-2: \t", payoff(level_2, opp_h, test_games))
# println("Maximin: \t", payoff(maximin, opp_h, test_games))
# println("Sum joint: \t",payoff(maxjoint, opp_h, test_games))
#
#
# println(@sprintf("PD for h: %0.f", decide(h, prisoners_dilemma)))
# # level_2 = SimHeuristic(level_1, level_1, 4.)
# # maximin = Heuristic(1,1,0,0,4, (+), minimum)
# # sum_prod = Heuristic(1,1,1,1,4, (*), sum)
# # max_prod = Heuristic(1,1,1,1,4, (*), maximum)
# # max_sum = Heuristic(1,1,1,1,4, (+), maximum)
# # sum_sum = Heuristic(1,1,1,1,4, (+), sum)
