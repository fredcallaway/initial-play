using Distributed
using Plots
import StatsBase: sample, std
using JSON
import Base: ==, hash
using DataFrames
using Distances

include("Heuristics.jl")

if length(workers()) == 1
    addprocs(Sys.CPU_THREADS)
    # addprocs(8)
end

@everywhere begin
    include("Heuristics.jl")
end


#%%
costs = Costs(0.07, 0.1, 0.3, 2.4)
opp_h = QLK(0.07, 0.64, 2.3)

games = [Game(3, -0.7) for i in 1:1000]
mh = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
# mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(mh))
mh = opt_prior!(mh, games, opp_h, costs)

opp_mh = deepcopy(mh)
mh = opt_prior!(mh, games, opp_mh, costs)


h_dist = [h_distribution(mh, g, opp_h, costs) for g in games]

mh
mean(h_dist)

## TODO: Fixa så att du kan spela mot en Meta Heuristic!!

#%% Calculating optimal strategy agains qlk for different ρ
@everywhere begin
    costs = Costs(0.08, 0.1, 0.2, 2.4)
    opp_h = QLK(0.07, 0.64, 2.3)
end

ρ_h_dists = pmap(-1:0.1:1) do ρ
    games = [Game(3, ρ) for i in 1:1000]
    mh = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
    mh = opt_prior!(mh, games, opp_h, costs)
    h_dists = [h_distribution(mh, g, opp_h, costs) for g in games]
    avg_h_dist = mean(h_dists)
    println(ρ, avg_h_dist)
    return(ρ, avg_h_dist)
end

#%% Find games where behaviors differ a lot
games = [Game(3, -0.8) for i in 1:1000]
mh_negative = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
# mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(mh))
mh_negative = opt_prior!(mh_negative, games, opp_h, costs)
h_dists = [h_distribution(mh_negative, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)


games = [Game(3, 0.8) for i in 1:1000]
mh_positive = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
# mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(mh))
mh_positive = opt_prior!(mh_positive, games, opp_h, costs)
h_dists = [h_distribution(mh_positive, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)

games = [Game(3, 0.) for i in 1:1000]

pos = play_distribution(mh_positive, games[1])
neg = play_distribution(mh_negative, games[1])

games_df = DataFrame(Game = Game[], pos = Vector{Float64}[], neg = Vector{Float64}[], div = Float64[])
games_df_old = games_df

games_df = DataFrame(Game = Game[], pos_row = Vector{Float64}[], neg_row = Vector{Float64}[], div_row = Float64[],  pos_col = Vector{Float64}[], neg_col = Vector{Float64}[], div_col = Float64[], tot_div = Float64[])
for i in 1:1000
        game = Game(3, 0.)
        pos_row = play_distribution(mh_positive, game)
        neg_row = play_distribution(mh_negative, game)
        row_div = kl_divergence(pos_row,neg_row)
        pos_col = play_distribution(mh_positive, transpose(game))
        neg_col = play_distribution(mh_negative, transpose(game))
        col_div = kl_divergence(pos_col,neg_col)
        tot_div = row_div + col_div
        push!(games_df, Dict(:Game => game, :pos_row => pos_row, :neg_row => neg_row, :div_row => row_div, :pos_col => pos_col, :neg_col => neg_col, :div_col => col_div, :tot_div => tot_div))
end

games_df
sort!(games_df, :tot_div, rev = true)

for i in 1:10
    println("-----")
    println(games_df.pos_row[i])
    println(games_df.neg_row[i])
    println(games_df.pos_col[i])
    println(games_df.neg_col[i])
    println(games_df.Game[i])
end

print(games_df.Game[1])


#%% Testing interesting games
games = [Game(3, -0.8) for i in 1:1000]
mh_negative = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
mh_negative = opt_prior!(mh_negative, games, opp_h, costs)
mh_negative_no = MetaHeuristic([RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0.])
mh_negative_no = opt_prior!(mh_negative_no, games, opp_h, costs)
# mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(mh))
h_dists = [h_distribution(mh_negative, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)


games = [Game(3, 0.8) for i in 1:1000]
mh_positive = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0., 0.])
mh_positive = opt_prior!(mh_positive, games, opp_h, costs)
mh_positive_no = MetaHeuristic([RowγHeuristic(2., 2.), RowγHeuristic(1., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-1., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(-0.2, 1.), RowHeuristic(-0.2, 2.)])], [0., 0., 0., 0., 0., 0.])
mh_positive_no = opt_prior!(mh_positive_no, games, opp_h, costs)
# mh = optimize_h!(mh, games, opp_h, costs; init_x = get_parameters(mh))
h_dists = [h_distribution(mh_positive, g, opp_h, costs) for g in games]
avg_h_dist = mean(h_dists)


games_dict = Dict()
games_dict["weak_link"] = Game([[8. 3. 0.]; [5. 5. 1.]; [4. 4. 4.]], [[8. 5. 4.]; [3. 5. 4.]; [0. 1. 4.]])
# games_dict["centipede"] = Game([[2 2 2]; [1 4 4]; [1 3 10]], [[0 0 0]; [3 1 1]; [3 7 4]])
games_dict["prisoners"] = Game([[8 2 1]; [9 3 1]; [1 0 1]], [[8 9 0]; [2 3 1]; [3 2 1]])
# games_dict["prisoners"] = Game([[9 7 0]; [8 8 1]; [5 6 3]], [[9 8 5]; [7 8 6]; [0 1 3]])
games_dict["travellers"] = Game([[2 4 4]; [0 3 5];[0 1 4]], [[2 0 0]; [4 3 1]; [4 5 4]])
games_dict["stag_hunt"] = Game([[9 4 0]; [6 6 1]; [4 4 4]], [[9 6 4]; [4 6 4]; [0 1 4]])
# games_dict["31"] = Game([[9 2 8];[5 5 6]; [6 0 3]], [[3 8 7]; [4 8 3]; [5 2 0]])
games_dict["sym"] = Game([[8 7 1]; [8 9 0]; [6 5 6]], [[8 8 4]; [7 9 5]; [1 0 6]])
games_dict["max"] = Game([[7 3 4]; [5 3 3]; [0 1 9]], [[4 5 0]; [3 7 2]; [3 3 9]])
for (name, game) in games_dict
    println(name)
    println(play_distribution(mh_positive, game, opp_h, costs))
    println(play_distribution(mh_negative, game, opp_h, costs))
    println(kl_divergence(play_distribution(mh_positive, game, opp_h, costs),play_distribution(mh_negative, game, opp_h, costs)))
    println(play_distribution(mh_positive, transpose(game), opp_h, costs))
    println(play_distribution(mh_negative, transpose(game), opp_h, costs))
    println(kl_divergence(play_distribution(mh_positive, transpose(game), opp_h, costs),play_distribution(mh_negative, transpose(game), opp_h, costs)))
    println(play_distribution(mh_positive_no, game, opp_h, costs))
    println(play_distribution(mh_negative_no, game, opp_h, costs))
    println(kl_divergence(play_distribution(mh_positive_no, game, opp_h, costs),play_distribution(mh_negative_no, game, opp_h, costs)))
    println(play_distribution(mh_positive_no, transpose(game), opp_h, costs))
    println(play_distribution(mh_negative_no, transpose(game), opp_h, costs))
    println(kl_divergence(play_distribution(mh_positive_no, transpose(game), opp_h, costs),play_distribution(mh_negative_no, transpose(game), opp_h, costs)))
    # println(play_distribution(mh_positive_no, game, opp_h, costs))
    # println(play_distribution(mh_negative_no, game))
    # println(play_distribution(mh_positive_no, transpose(game)))
    # println(play_distribution(mh_negative_no, transpose(game)))
end
