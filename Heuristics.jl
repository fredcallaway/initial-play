import Distributions: MvNormal
import Distributions
import StatsFuns: softmax
import StatsBase: sample, WeightVec
import Base
import DataStructures: OrderedDict
using BlackBoxOptim
import Printf: @printf

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
    λ::Float64
    cell_reduce::Function
    s_reduce::Function
end

mutable struct SimHeuristic
    opp_h::Heuristic
    self_h::Heuristic
    λ::Float64
end


Heuristic(dists::OrderedDict) = Heuristic(map(rand, dists.vals)...)
SimHeuristic(dists::OrderedDict) = SimHeuristic(Heuristic(dists), Heuristic(dists), rand(dists["λ"]))

function print_h(io::IO, h::Heuristic)
    # println("reduce_funs: ", h.cell_reduce, ", ", h.s_reduce)
    @printf(io, "Heuristic: funs(%s, %s) vars(β_r=%.2f, α_r=%.2f, β_c=%.2f, α_c=%.2f)", h.cell_reduce, h.s_reduce, h.β_r, h.α_r, h.β_c, h.α_c)
    # println("β_r=", round(h.β_r,3), " α_r=", h.α_r, " β_c=", h.β_c, " α_c=", h.α_c)
end
Base.show(io::IO, h::Heuristic) = @printf(io, "Heuristic: funs(%s, %s) vars(β_r=%.2f, α_r=%.2f, β_c=%.2f, α_c=%.2f)", h.cell_reduce, h.s_reduce, h.β_r, h.α_r, h.β_c, h.α_c)
Base.show(io::IO, s::SimHeuristic) = begin println("Opp_h: ",s.opp_h); println("Self_h: ",s.self_h); println("λ: ", s.λ) end
first_el = (x, y) -> x

h_dists = OrderedDict(
"β_r" => Distributions.Uniform(0,4),
"α_r" => Distributions.Uniform(0,4),
"β_c" => Distributions.Uniform(0,4),
"α_c" => Distributions.Uniform(0,4),
"λ" => Distributions.Uniform(0,4),
"cell_reduce" => [Base.:+, Base.:*, first_el],
"s_reduce" => [sum, maximum, minimum]
)

function relative_values(h::Heuristic, game::Game)
    v = zeros(size(game))
    for i in 1:size(game)
        r = game.row[i, :]
        c = game.col[i, :]
        r_tilde = map( x-> h.β_r * x^h.α_r, r)
        c_tilde = map( x-> h.β_c * x^h.α_c, c)
        s = map(h.cell_reduce, r_tilde, c_tilde)
        v[i] = h.s_reduce(s)
    end
    return v
end

function decide(h::Heuristic, game::Game)
    v = relative_values(h, game)
    v = softmax(h.λ*v)
    choice = argmax(v)
end

function decide(s::SimHeuristic, game::Game)
    g = deepcopy(game)
    opp_v = relative_values(s.opp_h, transpose(g))
    weights = softmax(s.λ*opp_v)
    g.row .*= weights'
    choice = decide(s.self_h, g)
    return choice
end



### Generating games and calcuating fitness
level_0 = Heuristic(0,0,0,0,0, (+), sum)
level_1 = Heuristic(1,1,0,0,0, (+), sum)
level_2 = SimHeuristic(level_1, level_1, 4.)
maximin = Heuristic(1,1,0,0,0, (+), minimum)
sum_prod = Heuristic(1,1,1,1,0, (*), sum)
max_prod = Heuristic(1,1,1,1,0, (*), maximum)
max_sum = Heuristic(1,1,1,1,0, (+), maximum)
sum_sum = Heuristic(1,1,1,1,0, (+), sum)


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


function opt_h!(h::Heuristic, opp_h::Heuristic, games; max_time=10.0)
    opp_plays = opp_cols_played(opp_h, games)
    param_names = [:β_r, :α_r, :β_c, :α_c]
    init_params = [h.β_r, h.α_r, h.β_c, h.α_c]
    function opt_fun(params)
        for (name, param) in zip(param_names, params)
            setfield!(h, name, param)
        end
        return -fitness_from_col_play(h, games, opp_plays)
    end
    # res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 4, MaxTime=10.0, Method = :separable_nes, TraceMode = :silent, TraceInterval =5.0)
    res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 4, MaxTime=max_time, TraceMode = :silent, TraceInterval =5.0)
    for (name, param) in zip(param_names, best_candidate(res))
        setfield!(h, name, param)
    end
end

function opt_s!(s::SimHeuristic, opp_h::Heuristic, games; max_time=10.0)
    opp_plays = opp_cols_played(opp_h, games)
    param_names = [(s.opp_h,:β_r), (s.opp_h,:α_r), (s.opp_h,:β_c), (s.opp_h,:α_c), (s.self_h,:β_r), (s.self_h, :α_r), (s.self_h, :β_c), (s.self_h, :α_c), (s,:λ)]
    init_params = [s.opp_h.β_r, s.opp_h.α_r, s.opp_h.β_c, s.opp_h.α_c, s.self_h.β_r, s.self_h.α_r, s.self_h.β_c, s.self_h.α_c, s.λ]
    function opt_fun(params)
        for (name, param) in zip(param_names, params)
            setfield!(name[1], name[2], param)
        end
        return -fitness_from_col_play(s, games, opp_plays)
    end
    # res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 9, MaxTime=10.0, Method = :separable_nes, TraceMode = :silent, TraceInterval =5.0)
    res = bboptimize(opt_fun; SearchRange = (h_dists["β_r"].a ,h_dists["β_r"].b), NumDimensions = 9, MaxTime=max_time, TraceMode = :silent, TraceInterval =5.0)
    for (name, param) in zip(param_names, best_candidate(res))
        setfield!(name[1], name[2], param)
    end
end


function find_best_heuristic(h_dists, opp_h, games)
    h_list = []
    # for cell_reduce in h_dists["cell_reduce"], s_reduce in h_dists["s_reduce"]
    #     h = Heuristic(h_dists)
    #     h.cell_reduce = cell_reduce
    #     h.s_reduce = s_reduce
    #     println("pre-fitnes: ", payoff(h, opp_h, games), " for ", h)
    #     opt_h!(h, opp_h, games)
    #     new_fitness = payoff(h, opp_h, games)
    #     println("post-fitnes: ", new_fitness, " for ", h)
    #     append!(h_list, [(h, new_fitness)])
    # end
    for opp_cell_reduce in h_dists["cell_reduce"], opp_s_reduce in h_dists["s_reduce"], self_cell_reduce in h_dists["cell_reduce"], self_s_reduce in h_dists["s_reduce"]
        s = SimHeuristic(h_dists)
        s.opp_h.cell_reduce = opp_cell_reduce
        s.opp_h.s_reduce = opp_s_reduce
        s.self_h.cell_reduce = self_cell_reduce
        s.self_h.s_reduce = self_s_reduce
        println("pre-fitnes: ", payoff(s, opp_h, games), " for ", s)
        opt_s!(s, opp_h, games)
        new_fitness = payoff(s, opp_h, games)
        println("post-fitnes: ", new_fitness, " for ", s)
        append!(h_list, [(s, new_fitness)])
    end
    sort!(s_list, rev=true, by= x -> x[2])
    return(s_list)
end




ρ = 0.
n_games = 1000
opp_h = level_1
println("Compare with ρ = ", ρ)
training_games = [Game(2, ρ) for i in range(1,n_games)];
test_games = [Game(2, ρ) for i in range(1,n_games)];

println("Level-0: \t", payoff(level_0, opp_h, training_games))
println("Level-1: \t", payoff(level_1, opp_h, training_games))
println("Level-2: \t", payoff(level_2, opp_h, training_games))
println("Maximin: \t", payoff(maximin, opp_h, training_games))
println("Sum prod: \t",payoff(sum_prod, opp_h, training_games))
println("Max prod: \t", payoff(max_prod, opp_h, training_games))
println("Max sum: \t", payoff(max_sum, opp_h, training_games))
println("Sum sum: \t", payoff(sum_sum, opp_h, training_games))


# println("0 0  ", payoff(level_0, level_0, training_games))
# println("0 1  ", payoff(level_0, level_1, training_games))
# println("1 0  ", payoff(level_1, level_0, training_games))
# println("1 1  ", payoff(level_1, level_1, training_games))

### Compare Heuristics
# heuristics = [Heuristic(h_dists) for i in 1:200]
# perf = map(h -> payoff(h, opp_h, training_games), heuristics)
# best_h = heuristics[argmax(perf)]
# println("Best (",maximum(perf), "):", heuristics[argmax(perf)])

h_list = find_best_heuristic(h_dists, opp_h, training_games)
best_h = h_list[1][1]
best_h_fit = h_list[1][2]
println("Best (",best_h_fit, "):", best_h)



println("Test games:")

println("Best h: \t", payoff(best_h, opp_h, test_games))
println("Level-0: \t", payoff(level_0, opp_h, test_games))
println("Level-1: \t", payoff(level_1, opp_h, test_games))
println("Level-2: \t", payoff(level_2, opp_h, test_games))
println("Maximin: \t", payoff(maximin, opp_h, test_games))
println("Sum prod: \t",payoff(sum_prod, opp_h, test_games))
println("Max prod: \t", payoff(max_prod, opp_h, test_games))
println("Max sum: \t", payoff(max_sum, opp_h, test_games))
println("Sum sum: \t", payoff(sum_sum, opp_h, test_games))


prisoners_dilemma = Game([[3 0];[4 1]], [[3 4]; [0 1]])
println("Prisoners dilemma test")
print(decide(best_h, prisoners_dilemma))


h = Heuristic(h_dists)
println("pre-fitnes: ", payoff(h, opp_h, training_games), " for ", h)
opt_h!(h, opp_h, training_games)
new_fitness = payoff(h, opp_h, training_games)
println("post-fitnes: ", new_fitness, " for ", h)
println("Prisoners dilemma test")
print(decide(best_h, prisoners_dilemma))

# Best :separable_nes -6820
