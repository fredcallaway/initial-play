using LinearAlgebra: norm
using StatsBase: entropy
using JSON
using CSV
using DataFrames
using DataFramesMeta
using Random
include("Heuristics.jl")


function json_to_game(s)
    a = JSON.parse(s)
    row = [convert(Float64, a[i][j][1]) for i in 1:length(a), j in 1:length(a[1])]
    col = [convert(Float64, a[i][j][2]) for i in 1:length(a), j in 1:length(a[1])]
    row_g = Game(row, col)
end
###########################################################################
#%% Load the data
###########################################################################

particapant_df = CSV.read("pilot/dataframes/participant_df.csv")
individal_choices_df = CSV.read("pilot/dataframes/individal_choices_df.csv")
# common_games_df = CSV.read("pilot/dataframes/positive_games_df.csv")
# competing_games_df = CSV.read("pilot/dataframes/negative_games_df.csv")
competing_games_df = CSV.read("data/processed/e3jydlve_play_distributions.csv");
common_games_df = CSV.read("data/processed/nlesp5ss_play_distributions.csv");
# comparison_idx = [31, 37, 41, 44, 49]

comparison_idx = [31, 34, 38, 41, 44, 50]


parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))
function games_and_plays(df)
    row_plays = map(eachrow(df)) do x
        game = json_to_game(x.row_game)
        game.row[1,2] += 1e-5
        game.col[1,2] += 1e-5
        play_dist = parse_play(x.row_play)
        (game, play_dist)
    end
    col_plays = map(eachrow(df)) do x
        game = json_to_game(x.col_game)
        game.row[2,1] += 1e-5
        game.col[2,1] += 1e-5
        # break_symmetry!(game)
        play_dist = parse_play(x.col_play)
        (game, play_dist)
    end
    result = append!(row_plays, col_plays)
    result
end


positive_data = games_and_plays(common_games_df)
negative_data = games_and_plays(competing_games_df)

data_dicts = Dict()
data_dicts["positive"] = Dict("row" => [Dict("game"=> positive_data[i][1], "play"=> positive_data[i][2]) for i in 1:50],
                              "col" => [Dict("game"=> positive_data[i+50][1], "play"=> positive_data[i+50][2]) for i in 1:50])
data_dicts["negative"] = Dict("row" => [Dict("game"=> negative_data[i][1], "play"=> negative_data[i][2]) for i in 1:50],
                              "col" => [Dict("game"=> negative_data[i+50][1], "play"=> negative_data[i+50][2]) for i in 1:50])

# data_dicts["positive"] = Dict("row" => [Dict("game"=> json_to_game(first(common_games_df[common_games_df.round .== i, :row])), "play"=> convert(Vector{Float64}, JSON.parse(first(common_games_df[common_games_df.round .== i, :row_play])))) for i in 1:50],
#                               "col" => [Dict("game"=> json_to_game(first(common_games_df[common_games_df.round .== i, :col])), "play"=> convert(Vector{Float64}, JSON.parse(first(common_games_df[common_games_df.round .== i, :col_play])))) for i in 1:50])
# data_dicts["negative"] = Dict("row" => [Dict("game"=> json_to_game(first(competing_games_df[competing_games_df.round .== i, :row])), "play"=> convert(Vector{Float64}, JSON.parse(first(competing_games_df[competing_games_df.round .== i, :row_play])))) for i in 1:50],
#                               "col" => [Dict("game"=> json_to_game(first(competing_games_df[competing_games_df.round .== i, :col])), "play"=> convert(Vector{Float64}, JSON.parse(first(competing_games_df[competing_games_df.round .== i, :col_play])))) for i in 1:50])


opp_h_dict = Dict("positive" => Dict(), "negative" => Dict())
opp_h_dict["positive"]["row"] = CacheHeuristic([x["game"] for x in data_dicts["positive"]["row"]], [x["play"] for x in data_dicts["positive"]["row"]])
opp_h_dict["positive"]["col"] = CacheHeuristic([x["game"] for x in data_dicts["positive"]["col"]], [x["play"] for x in data_dicts["positive"]["col"]])
opp_h_dict["negative"]["row"] = CacheHeuristic([x["game"] for x in data_dicts["negative"]["row"]], [x["play"] for x in data_dicts["negative"]["row"]])
opp_h_dict["negative"]["col"] = CacheHeuristic([x["game"] for x in data_dicts["negative"]["col"]], [x["play"] for x in data_dicts["negative"]["col"]])


######################################################################
#%% Acutal Rule Learning
########################################################################

function rule_loss(mh_init, β₀, β₁, costs; treats = ["positive", "negative"], roles = ["row", "col"])
    pred_loss = 0
    for treat in treats, role in roles
        mh = deepcopy(mh_init)
        opp_role = role == "row" ? "col" : "row"
        opp_h = opp_h_dict[treat][opp_role]
        actual_h = opp_h_dict[treat][role]
        for r in 1:50
            game = data_dicts[treat][role][r]["game"]
            for i in 1:length(mh.h_list)
                mh.prior[i] = β₀ * mh.prior[i] + β₁ * perf(mh.h_list[i], [game], opp_h, costs)
            end
            # pred_loss = pred_loss + prediction_loss_no_RI(mh, [game], actual_h, opp_h, costs)*15 # This is to get sum of loss instead of average
            pred_loss = pred_loss + prediction_loss(mh, [game], actual_h, opp_h, costs)*15 # This is to get sum of loss instead of average
        end
    end
    return pred_loss
end

function end_rules(mh_init, β₀, β₁, costs; treats = ["positive", "negative"], roles = ["row", "col"])
    rules = Dict("positive" => Dict(), "negative" => Dict())
    for treat in treats, role in roles
        mh = deepcopy(mh_init)
        opp_role = role == "row" ? "col" : "row"
        opp_h = opp_h_dict[treat][opp_role]
        actual_h = opp_h_dict[treat][role]
        for r in 1:50
            game = data_dicts[treat][role][r]["game"]
            for i in 1:length(mh.h_list)
                mh.prior[i] = β₀ * mh.prior[i] + β₁ * perf(mh.h_list[i], [game], opp_h, costs)
            end
        end
        rules[treat][role] = mh
    end
    return rules
end

# To do stuff with
function prediction_loss_no_RI(h::MetaHeuristic, games::Vector{Game}, actual::Heuristic, costs::Costs; loss_f = likelihood)
    loss = 0
    for game in games
        pred_p = play_distribution(h, game)
        actual_p = play_distribution(actual, game)
        loss += loss_f(pred_p, actual_p)
    end
    loss/length(games)
end

function optimize_rule_params(mh_init, β₀, β₁, costs_init; treats = ["positive", "negative"], roles=["row", "col"])
    mh = deepcopy(mh_init)
    costs = deepcopy(costs_init)
    # init_x = [β₀, β₁, costs.α, costs.λ, costs.level, costs.m_λ, mh.prior...]
    init_x = [β₀, β₁, costs.α, costs.λ, costs.level, mh.prior...]
    # init_x = [β₀, β₁, costs.α, costs.λ, costs.level, costs.m_λ]
    function loss_f(x)
        β₀ = x[1]
        β₁ = x[2]
        costs.λ = x[3]
        costs.α = x[4]
        costs.level = x[5]
        # costs.m_λ = x[6]
        # mh.prior = x[7:end]
        mh.prior = x[6:end]
        return rule_loss(mh, β₀, β₁, costs; treats = treats, roles=roles)
    end
    res_x = Optim.minimizer(optimize(loss_f, init_x))
    β₀ = res_x[1]
    β₁ = res_x[2]
    costs.λ = res_x[3]
    costs.α = res_x[4]
    costs.level = res_x[5]
    # costs.m_λ = res_x[6]
    mh.prior = res_x[6:end]
    return (β₀, β₁, costs, mh)
end

function get_lambdas(h::Heuristic)
    return [h.λ]
end

function set_lambdas!(h::Heuristic, x::Vector{T} where T <: Real)
    h.λ = x[1]
end

function get_lambdas(h::RandomHeuristic)
    return []
end

function set_lambdas!(h::RandomHeuristic, x::Vector{T} where T <: Real)
    pass
end

function get_lambdas(h::SimHeuristic)
    [h.λ for h in h.h_list]
end

function set_lambdas!(sh::SimHeuristic, x::Vector{T} where T <: Real)
    for (h, val) in zip(sh.h_list, x)
        h.λ = val
    end
end

function get_lambdas(mh::MetaHeuristic)
    [x for h in mh.h_list for x in get_lambdas(h)]
end


function set_lambdas!(mh::MetaHeuristic, x::Vector{T} where T <: Real)
    idx = 1
    for h in mh.h_list
        set_lambdas!(h, x[idx:(idx+length(get_lambdas(h)) -1)])
        idx = idx+length(get_lambdas(h))
    end
    mh
end

function optimize_rule_lambdas(mh_init, β₀, β₁, costs; treats = ["positive", "negative"], roles=["row", "col"])
    mh = deepcopy(mh_init)
    init_x = get_lambdas(mh)
    function loss_f(x)
        set_lambdas!(mh, x)
        return rule_loss(mh, β₀, β₁, costs; treats = treats, roles=roles)
    end
    res_x = Optim.minimizer(optimize(loss_f, init_x))
    mh = set_lambdas!(mh, res_x)
    return mh
end

costs = Costs(0.2, 0.2, 0.1, 1.)
mh_default = MetaHeuristic([JointMax(2.), RowγHeuristic(2., 2.), RowγHeuristic(0., 2.), RowγHeuristic(-2., 2.), SimHeuristic([RowHeuristic(0., 2.), RowHeuristic(0., 2.)])], [0., 0., 0., 0., 0.]);



mh = deepcopy(mh_default)
β₀ = 0.5
β₁ = 0.5
res = optimize_rule_params(mh, β₀, β₁, costs)

β₀, β₁, costs, mh = res

print(game)
play_distribution(mh, game)

# rule_loss(mh, β₀, β₁, costs)/(60*50)

mh = optimize_rule_lambdas(mh, 0.5, 0.5, costs)

mh

rule_loss(mh, β₀, β₁, costs; treats = ["positive"])/(30*50)
rule_loss(mh, β₀, β₁, costs; treats = ["negative"])/(30*50)

rules = end_rules(mh_default, β₀, β₁, costs)
rules["positive"]["row"]
rules["positive"]["col"]
rules["negative"]["row"]
rules["negative"]["col"]
