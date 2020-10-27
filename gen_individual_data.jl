using DataFrames
using DataFramesMeta
using CSV
using JSON
using StatsBase
using SplitApplyCombine

include("Heuristics.jl")

function write_ind_data(df_wide_all, session_code)
    # names!(df_wide_all, map(n->Symbol(replace(string(n), "." => "_")), names(df_wide_all)))
    n_page = maximum(df_wide_all.participant__index_in_pages)
    keep = ((df_wide_all.participant__index_in_pages .== n_page) .&
            (df_wide_all.session_code .== session_code))
    df_wide = df_wide_all[keep, :]

    participant_df = map(eachrow(df_wide)) do row
        (pid = row.participant_code,
         started = row.participant_time_started,
         tot_payoff = row.participant_payoff,
         treatment = row.normal_form_games_1_player_treatment,
         session_code = row.session_code)
    end |> DataFrame



    getvar(name, i) = Symbol("normal_form_games_" * string(i) * "_player_" * string(name))

    individal_choices_df = mapmany(eachrow(df_wide)) do row
        map(1:50) do i
            (session_code = row.session_code,
             pid = row.participant_code,
             treatment = row[getvar(:treatment, i)],
             role = row[getvar(:player_role, i)],
             round = i,
             choice = row[getvar(:choice, i)],
             other_choice = row[getvar(:other_choice, i)])
        end
    end

    #Bugfix, for some reason I had to do this even though it worked before as with participant_df | Gustav 2019-08-27
    individal_choices_df = convert(Array{typeof(individal_choices_df[1])}, individal_choices_df) |> DataFrame

    treatment = individal_choices_df.treatment[1]


    x = by(individal_choices_df, [:round, :role],
        :choice => x -> Tuple(counts(x, 0:2) ./ length(x))
    )

    df = unstack(x, :role, :choice_function)
    rename!(df, :col => :col_play, :row => :row_play)

    row = []
    col = []
    for i in 1:50
        games = df_wide[getvar(:game, i)]
        col_idx = argmax(df_wide[getvar(:player_role, i)] .== "col")
        row_idx = argmax(df_wide[getvar(:player_role, i)] .== "row")
        push!(row, games[row_idx])
        push!(col, games[col_idx])
    end


    comparison = [31, 38, 42, 49]
    get_row_game(x) = row[x]
    get_col_game(x) = col[x]
    get_row_play(x) = df.row_play[x]
    get_col_play(x) = df.col_play[x]
    get_game(x, role) = role == "row" ? row[x] : col[x]
    get_type(x) = x in comparison ? "comparison" : "treatment"
    individal_choices_df = @transform(individal_choices_df, row_game_string = get_row_game.(:round), col_game_string =get_col_game(:round), game_string=get_game.(:round,:role), type= get_type.(:round), row_play = get_row_play.(:round), col_play=get_col_play.(:round))
    individal_choices_df |> CSV.write("data/processed/ind_data/$(treatment)/$(session_code)_ind_play_distributions.csv")
end

# raw_csv = "data/pilot/all_apps_wide_2019-04-06.csv"
# raw_csv = "data/raw/all_apps_wide_2019-05-03.csv"
# raw_csv = "data/raw/all_apps_wide_2019-08-22.csv"
raw_csv = "data/raw/all_apps_wide_2019-09-01.csv"

df_wide_all = CSV.read(raw_csv);
names!(df_wide_all, map(n->Symbol(replace(string(n), "." => "_")), names(df_wide_all)));
collect(countmap(df_wide_all.session_code))


for (code,n) in countmap(df_wide_all.session_code)
    if n >= 30
        println(write_ind_data(df_wide_all, code))
    end
end
