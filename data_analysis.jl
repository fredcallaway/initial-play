using DataFrames
using CSV
using JSON
include("Heuristics.jl")


df_wide_all = CSV.read("Gustav_pilot/all_apps_wide_2019-04-04.csv")

df_wide = df_wide_all[df_wide_all.participant__index_in_pages .== 255, :]

participant_df = DataFrame()
for row in eachrow(df_wide)
    dict = Dict()
    dict[:pid] = row.participant_code
    dict[:started] = row.participant_time_started
    dict[:tot_payoff] = row.participant_payoff
    dict[:treatment] = row.normal_form_games_1_player_treatment
    dict[:role] = row.normal_form_games_1_player_player_role
    dict[:session_code] = row.session_code
    if length(names(participant_df)) == 0
        participant_df = DataFrame(dict)
    elseif  ! (dict[:pid] in participant_df.pid)
        push!(participant_df, dict)
    end
end

individal_choices_df = DataFrame()
for row in eachrow(df_wide)
    pid = row.participant_code
    treatment = row.normal_form_games_1_player_treatment
    role = row.normal_form_games_1_player_player_role
    session = row.session_code
    for i in 1:50
        dict = Dict()
        dict[:session_code] = session
        dict[:pid] = pid
        dict[:treatment] = treatment
        dict[:role] = role
        dict[:round] = i
        dict[:choice] = row[Symbol("normal_form_games_" * string(i) * "_player_choice")]
        dict[:other_choice] = row[Symbol("normal_form_games_" * string(i) * "_player_other_choice")]
        if length(names(individal_choices_df)) == 0
            individal_choices_df = DataFrame(dict)
        elseif  ! any(all.(zip(individal_choices_df.pid .== dict[:pid], individal_choices_df.round .== i)))
            push!(individal_choices_df, dict)
        end
    end
end



function json_to_game(s)
    a = JSON.parse(s)
    row = [convert(Float64, a[i][j][1]) for i in 1:length(a), j in 1:length(a[1])]
    col = [convert(Float64, a[i][j][2]) for i in 1:length(a), j in 1:length(a[1])]
    row_g = Game(row, col)
end
# for row in eachrow(df_wide)
# for treatment in ["positive"]
positive_games_df = DataFrame()
comparison_games = [31, 37, 41, 44, 49]
# for treatment in ["positive", "negative"]
for i in 1:50
    treat = "positive"
    dict = Dict()
    dict[:session_code] = first(df_wide.session_code)
    dict[:row] = first(df_wide[(df_wide.normal_form_games_1_player_treatment .== treat) .& (df_wide.normal_form_games_1_player_player_role .== "row"), Symbol("normal_form_games_"*string(i)*"_player_game")])
    dict[:col] = first(df_wide[(df_wide.normal_form_games_1_player_treatment .== treat) .& (df_wide.normal_form_games_1_player_player_role .== "row"), Symbol("normal_form_games_"*string(i)*"_player_game")])
    dict[:round] = i
    dict[:type] = i in comparison_games ? "comparison" : "treatment"
    play_dists = Dict("row" => zeros(3), "col" => zeros(3))
    for row in eachrow(individal_choices_df[individal_choices_df.round .== i, [:role, :choice]])
        play_dists[row.role][row.choice+1] += 1
    end
    dict[:row_play] = JSON.json((play_dists["row"]/sum(play_dists["row"])))
    dict[:col_play] = JSON.json(play_dists["col"]/sum(play_dists["col"]))
    dict
    if length(names(positive_games_df)) == 0
        positive_games_df = DataFrame(dict)
    elseif length(positive_games_df[positive_games_df.round .== i, :round]) == 0
        push!(positive_games_df, dict)
    end
end

CSV.write("Gustav_pilot/dataframes/participant_df.csv", participant_df)
CSV.write("Gustav_pilot/dataframes/individal_choices_df.csv", individal_choices_df)
CSV.write("Gustav_pilot/dataframes/positive_games_df.csv", positive_games_df)
