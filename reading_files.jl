using DataFrames
using CSV
include("PHeuristics.jl")


asymmetric_games = CSV.read("data/asymmetric_games.csv")
symmetric_games = CSV.read("data/symmetric_games.csv")


sort!(symmetric_games, (order(:GAME), order(:ACTION)))
sym_games_list = []
for name in unique(symmetric_games[:, :GAME])
        game = symmetric_games[symmetric_games.GAME .== name,:]
        plays = game[:,:COUNT]
        plays = plays/sum(plays)
        data_set = game[1, :DATASET]
        game_obj = get_game(name)
        test_dict = Dict("name" => name, "game" => game_obj, "plays" => plays, "data_set" => data_set)
        push!(sym_games_list, test_dict)
end

sort!(asymmetric_games, (order(:GAME), order(:PLAYER), order(:ACTION)))
asym_games_list = []
for name in unique(asymmetric_games[:, :GAME])
        game = asymmetric_games[asymmetric_games.GAME .== name,:]
        p1_plays = game[game.PLAYER .== 1,:COUNT]
        p1_plays = p1_plays/sum(p1_plays)
        p2_plays = game[game.PLAYER .== 2,:COUNT]
        p2_plays = p2_plays/sum(p2_plays)
        data_set = game[1, :DATASET]
        game_obj = get_game(name)
        test_dict = Dict("name" => name, "game" => game_obj, "p1_plays" => p1_plays, "p2_plays" => p2_plays, "data_set" => data_set)
        push!(asym_games_list, test_dict)
end

asym_games_list[1]["name"]

function get_game(g_name)
        g_name = replace(lowercase(g_name), "-" => "_")
        game = nothing
        open("data/"*g_name*".nfg") do f
                lines = readlines(f)
                n_strats = replace(lines[2], r"\".\"" => "")
                n_strats = replace(n_strats, r"{|}" => "")
                p1_n, p2_n = strip(n_strats) |> x -> split(x, " ") |> x -> (parse(Int64, x[1]), parse(Int64, x[2]))
                if p1_n == p2_n
                        payoff_list = map(x -> parse(Float64, x), split(strip(lines[3])," "))
                        p1_payoffs = payoff_list[filter(x -> x % 2 == 1, 1:length(payoff_list))]
                        p2_payoffs = payoff_list[filter(x -> x % 2 == 0, 1:length(payoff_list))]
                        game = Game(reshape(p1_payoffs, (p1_n, p2_n)), reshape(p2_payoffs, (p1_n, p2_n)))
                end
        end
        return game
end


for dict in sym_games_list
        g_name = dict["game"]
        game = get_game(g_name)
        if game != nothing
                println(get_game(g_name))
        end
end



get_game("CGCB2b")

g_name = "cgcb2b"
open("data/"*g_name*".nfg") do f
        lines = readlines(f)
        if length(lines) != 3
                println(lines)
        end
        println(lines[2])
        # n_strats = replace(lines[2], "{ \"0\" \"1\" }" => "")
        n_strats = replace(lines[2], r"\".\"" => "")
        n_strats = replace(n_strats, r"{|}" => "")
        p1_n, p2_n = strip(n_strats) |> x -> split(x, " ") |> x -> (parse(Int64, x[1]), parse(Int64, x[2]))
        if p1_n == p2_n
                payoff_list = map(x -> parse(Float64, x), split(strip(lines[3])," "))
                p1_payoffs = payoff_list[filter(x -> x % 2 == 1, 1:length(payoff_list))]
                p2_payoffs = payoff_list[filter(x -> x % 2 == 0, 1:length(payoff_list))]
                game = Game(reshape(p1_payoffs, (p1_n, p2_n)), reshape(p2_payoffs, (p1_n, p2_n)))
        end
end

replace("{ \"0\" \"1\" }", r"\".\"" => "")

asym_games_list[1]
