using DataFrames
using CSV
using JSON
using StatsBase
using SplitApplyCombine

include("Heuristics.jl")

function write_play_distributions(df_wide_all, session_code)
    names!(df_wide_all, map(n->Symbol(replace(string(n), "." => "_")), names(df_wide_all)))

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
    end |> DataFrame

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
        push!(row, games[col_idx])
        push!(col, games[row_idx])
    end

    df[:row_game] = row
    df[:col_game] = col
    df[:treatment] = participant_df.treatment[1]

    comparison = [31, 34, 38, 41, 44, 50]
    df[:type] = map(1:50) do i
        i in comparison ? "comparison" : "treatment"
    end

    df |> CSV.write("data/processed/$(session_code)_play_distributions.csv")
end

raw_csv = "data/raw/all_apps_wide_2019-05-03.csv"
df_wide_all = CSV.read(raw_csv);
names!(df_wide_all, map(n->Symbol(replace(string(n), "." => "_")), names(df_wide_all)));
for (code,n) in countmap(df_wide_all.session_code)
    if n >= 30
        println(write_play_distributions(df_wide_all, code))
    end
end
