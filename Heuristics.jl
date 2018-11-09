import Distributions: MvNormal
import Distributions
import Base
import DataStructures: OrderedDict
using BlackBoxOptim

function sample_cell(ρ; max=10, min=0, µ=5, σ=5)
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
    β_r::Float64
    α_r::Float64
    β_c::Float64
    α_c::Float64
    cell_reduce::Function
    # α_s::Float64
    s_reduce::Function
end

Heuristic(dists::OrderedDict) = Heuristic(map(rand, dists.vals)...)

first_el = (x, y) -> x

h_dists = OrderedDict(
"β_r" => Distributions.Uniform(0,4),
"α_r" => Distributions.Uniform(0,4),
"β_c" => Distributions.Uniform(0,4),
"α_c" => Distributions.Uniform(0,4),
"cell_reduce" => [Base.:+, Base.:*, Base.:/, first_el],
# "α_s" => Distributions.Uniform(0,4),
"s_reduce" => [sum, maximum, minimum]
)

function decide(h::Heuristic, game::Game)
    v = zeros(size(game))
    for i in 1:size(game)
        r = game.row[i, :]
        c = game.col[i, :]
        r_tilde = map( x-> h.β_r * x^h.α_r, r)
        c_tilde = map( x-> h.β_c * x^h.α_c, c)
        s = map(h.cell_reduce, r_tilde, c_tilde)
        # s_tilde = map( x-> x^h.α_s, s) TODO: Remeber that you did this!!
        s_tilde = map( x-> x, s)
        v[i] = h.s_reduce(s_tilde)
    end
    choice = argmax(v)
end

function mutate_heuristic(h, p, h_dists)
    for name in fieldnames(Heuristic)
        if rand() < p
            println("mutating")
            setfield!(h, name, rand(h_dists["$name"]))
        end
    end
end





### Generating games and calcuating fitness
level_0 = Heuristic(0,0,0,0, (+), sum)
level_1 = Heuristic(1,1,0,0, (+), sum)
maximin = Heuristic(1,1,0,0, (+), minimum)
sum_prod = Heuristic(1,1,1,1, (*), sum)
max_prod = Heuristic(1,1,1,1, (*), maximum)
max_sum = Heuristic(1,1,1,1, (+), maximum)
sum_sum = Heuristic(1,1,1,1, (+), sum)


function fitness(h, opp_h, games)
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
        decision = decide(h,games[i])
        payoff += opp_plays[i][decision]
    end
    return payoff
end


function opt_h!(h, opp_h, games)
    opp_plays = opp_cols_played(opp_h, training_games)
    param_names = [:β_r, :α_r, :β_c, :α_c]
    init_params = [h.β_r, h.α_r, h.β_c, h.α_c]
    function opt_fun(params)
        for (name, param) in zip(param_names, params)
            setfield!(h, name, param)
        end
        return -fitness_from_col_play(h, training_games, opp_plays)
    end
    # res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 4, MaxTime=10.0, Method = :separable_nes, TraceMode = :silent, TraceInterval =5.0)
    res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 4, MaxTime=10.0, TraceMode = :silent, TraceInterval =5.0)
    for (name, param) in zip(param_names, best_candidate(res))
        setfield!(h, name, param)
    end
end

function find_best_heuristic(h_dists, opp_h, games)
    h_list = []
    for cell_reduce in h_dists["cell_reduce"], s_reduce in h_dists["s_reduce"]
        h = Heuristic(h_dists)
        h.cell_reduce = cell_reduce
        h.s_reduce = s_reduce
        println("pre-fitnes: ", fitness(h, opp_h, games), " for ", h)
        opt_h!(h, opp_h, training_games)
        new_fitness = fitness(h, opp_h, games)
        println("post-fitnes: ", new_fitness, " for ", h)
        append!(h_list, [(h, new_fitness)])
    end
    sort!(h_list, rev=true, by= x -> x[2])
    return(h_list)
end




ρ = 0.
n_games = 1000
opp_h = level_1
println("Compare with ρ = ", ρ)
training_games = [Game(2, ρ) for i in range(1,n_games)];
test_games = [Game(2, ρ) for i in range(1,n_games)];

println("Level-0: \t", fitness(level_0, opp_h, training_games))
println("Level-1: \t", fitness(level_1, opp_h, training_games))
println("Maximin: \t", fitness(maximin, opp_h, training_games))
println("Sum prod: \t",fitness(sum_prod, opp_h, training_games))
println("Max prod: \t", fitness(max_prod, opp_h, training_games))
println("Max sum: \t", fitness(max_sum, opp_h, training_games))
println("Sum sum: \t", fitness(sum_sum, opp_h, training_games))


# println("0 0  ", fitness(level_0, level_0, training_games))
# println("0 1  ", fitness(level_0, level_1, training_games))
# println("1 0  ", fitness(level_1, level_0, training_games))
# println("1 1  ", fitness(level_1, level_1, training_games))

### Compare Heuristics
# heuristics = [Heuristic(h_dists) for i in 1:200]
# perf = map(h -> fitness(h, opp_h, training_games), heuristics)
# best_h = heuristics[argmax(perf)]
# println("Best (",maximum(perf), "):", heuristics[argmax(perf)])

h_list = find_best_heuristic(h_dists, opp_h, training_games)
best_h = h_list[1][1]
best_h_fit = h_list[1][2]
println("Best (",best_h_fit, "):", best_h)



println("Test games:")

println("Best h: \t", fitness(best_h, opp_h, test_games))
println("Level-0: \t", fitness(level_0, opp_h, test_games))
println("Level-1: \t", fitness(level_1, opp_h, test_games))
println("Maximin: \t", fitness(maximin, opp_h, test_games))
println("Sum prod: \t",fitness(sum_prod, opp_h, test_games))
println("Max prod: \t", fitness(max_prod, opp_h, test_games))
println("Max sum: \t", fitness(max_sum, opp_h, test_games))
println("Sum sum: \t", fitness(sum_sum, opp_h, test_games))


prisoners_dilemma = Game([[3 0];[4 1]], [[3 4]; [0 1]])
println("Prisoners dilemma test")
print(decide(best_h, prisoners_dilemma))


h = Heuristic(h_dists)
println("pre-fitnes: ", fitness(h, opp_h, training_games), " for ", h)
opt_h!(h, opp_h, training_games)
new_fitness = fitness(h, opp_h, training_games)
println("post-fitnes: ", new_fitness, " for ", h)
println("Prisoners dilemma test")
print(decide(best_h, prisoners_dilemma))

# Best :separable_nes -6820
