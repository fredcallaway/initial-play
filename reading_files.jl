using DataFrames
using CSV
using JSON
include("PHeuristics.jl")

asymmetric_games = CSV.read("data/asymmetric_games.csv")
symmetric_games = CSV.read("data/symmetric_games.csv")


function get_game(g_name)
        g_name = replace(lowercase(g_name), "-" => "_")
        game = nothing
        open("data/"*g_name*".nfg") do f
                lines = readlines(f)
                n_strats = replace(lines[2], r"\".\"" => "")
                n_strats = replace(n_strats, r"{|}" => "")
                p1_n, p2_n = strip(n_strats) |> x -> split(x, " ") |> x -> (parse(Int64, x[1]), parse(Int64, x[2]))
                # At the moment our code can only handle symmetric_games, few games are assymetric
                if p1_n == p2_n
                        payoff_list = map(x -> parse(Float64, x), split(strip(lines[3])," "))
                        p1_payoffs = payoff_list[filter(x -> x % 2 == 1, 1:length(payoff_list))]
                        p2_payoffs = payoff_list[filter(x -> x % 2 == 0, 1:length(payoff_list))]
                        game = Game(reshape(p1_payoffs, (p1_n, p2_n)), reshape(p2_payoffs, (p1_n, p2_n)))
                        if maximum(game.row) > 10
                                game.row ./= 10
                                game.col ./= 10
                        end
                end
        end
        return game
end

sort!(symmetric_games, (order(:GAME), order(:ACTION)))
sym_games_list = []
for name in unique(symmetric_games[:, :GAME])
        game = symmetric_games[symmetric_games.GAME .== name,:]
        plays = game[:,:COUNT]
        plays = plays/sum(plays)
        data_set = game[1, :DATASET]
        game_obj = get_game(name)
        if game_obj != nothing
                test_dict = Dict("name" => name, "game" => game_obj, "plays" => plays, "data_set" => data_set)
                push!(sym_games_list, test_dict)
        end
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
        if game_obj != nothing
                test_dict = Dict("name" => name, "game" => game_obj, "p1_plays" => p1_plays, "p2_plays" => p2_plays, "data_set" => data_set)
                push!(asym_games_list, test_dict)
        end
end


games = [x["game"] for x in sym_games_list]
opp_cols_played = [x["plays"] for x in sym_games_list]
self_rows_played = [x["plays"] for x in sym_games_list]


data_set_vec = [x["data_set"] for x in sym_games_list]
game_names_vec = [x["name"] for x in sym_games_list]

for dict in asym_games_list
        push!(games, dict["game"])
        push!(games, transpose(dict["game"]))
        push!(opp_cols_played, dict["p2_plays"])
        push!(opp_cols_played, dict["p1_plays"])
        push!(self_rows_played, dict["p1_plays"])
        push!(self_rows_played, dict["p2_plays"])
        push!(data_set_vec, dict["data_set"])
        push!(data_set_vec, dict["data_set"])
        push!(game_names_vec, dict["name"])
        push!(game_names_vec, dict["name"])
end


game_names_vec

# json_games = JSON.json(games)
#
# json_cols_plays = JSON.json(opp_cols_played)
#
# plays_vec_from_json(json_cols_plays)
#
# loaded_cols_played = JSON.parse(json_cols_plays)
#
# convert(Array{Array{Float64,1},1}, loaded_cols_played)
#
# loaded_cols_played

open("data/games.json", "w") do f
        json_games = JSON.json(games)
        write(f, json_games)
end

open("data/rows_played.json", "w") do f
        json_rows_played = JSON.json(self_rows_played)
        write(f, json_rows_played)
end


open("data/cols_played.json", "w") do f
        json_cols_played = JSON.json(opp_cols_played)
        write(f, json_cols_played)
end

open("data/dataset_list.json", "w") do f
        json_datasets = JSON.json(data_set_vec)
        write(f, json_datasets)
end

open("data/game_names_list.json", "w") do f
        json_names = JSON.json(game_names_vec)
        write(f, json_names)
end

#%%

games_from_json("data/games.json")[1]

function games_from_json(file_name)
        games_json = ""
        open(file_name) do f
                games_json = read(f, String)
        end
        games_vec = []
        games_json = JSON.parse(games_json)
        for g in games_json
                row = [g["row"][j][i] for i in 1:length(g["row"][1]), j in 1:length(g["row"])]
                col = [g["col"][j][i] for i in 1:length(g["col"][1]), j in 1:length(g["col"])]
                game = Game(row, col)
                push!(games_vec, game)
        end
        return games_vec
end
plays_vec_from_json("data/cols_played.json")
plays_vec_from_json("data/rows_played.json")

vector_list
vector_list = open("data/dataset_list.json") do f
        vector_list = JSON.parse(read(f, String))
end


vector_list
