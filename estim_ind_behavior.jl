using Flux
using JSON
using CSV
using DataFrames
using DataFramesMeta
using SplitApplyCombine
using Random
using Glob
using Distributed
# using BSON
using Serialization
using StatsBase
using Statistics
using Sobol
using Profile
using Tables





nprocs() == 1 && addprocs(Sys.CPU_THREADS - 1)
include("Heuristics.jl")  # prevent LoadError: UndefVarError: Game not defined below
@everywhere using ForwardDiff
@everywhere using DataFrames
@everywhere include("Heuristics.jl")
@everywhere include("model.jl")
#%%

@everywhere begin
    Data = Array{Tuple{Game,Array{Float64,1}},1}
    parse_play(x) = float.(JSON.parse(replace(replace(x, ")" => "]"),  "(" => "[",)))

    function load_data(file)
        df = DataFrame!(CSV.File(file));
        row_plays = Data()
        col_plays = Data()
        for row in eachrow(df)
            row = collect(eachrow(df))[1]
            row_game = json_to_game(row.row_game)
            row_game.row[1,2] += rand()*1e-7 # This can't be a constant number if we want to
            row_game.col[1,2] += rand()*1e-7 # separate behavior in comparison games in different treatments.
            row_play_dist = parse_play(row.row_play)
            col_game = transpose(row_game)
            col_play_dist = parse_play(row.col_play)
            push!(row_plays, (row_game, row_play_dist))
            push!(col_plays, (col_game, col_play_dist))
        end
        append!(row_plays, col_plays)
    end

    function play_from_int(x)
        s = zeros(3)
        if x > 0
            s[x] = 1.
        end
        s
    end


    function load_ind_treat(file)
        df = DataFrame!(CSV.File(file));
        rng = MersenneTwister(Int(hash(file)))
        row_noise = rand(rng,50) .*1e-9
        col_noise = rand(rng,50) .*1e-9
        new_df = map(rowtable(df)) do row
            game = json_to_game(row.game_string)
            row_game = json_to_game(row.row_game_string)
            col_game = json_to_game(row.col_game_string)
            game.row[1,2] += rand().*1e-9
            game.col[1,2] += rand().*1e-9
            row_game.row[1,2] += row_noise[row.round]
            row_game.col[1,2] += col_noise[row.round]
            col_game = transpose(row_game)
            # col_game.row[2,1] += row_noise[row.round]
            # col_game.col[2,1] += col_noise[row.round]

            row_play_dist = parse_play(row.row_play)
            col_play_dist = parse_play(row.col_play)
            opp_dist = row.role == "row" ? col_play_dist : row_play_dist
            self_dist = row.role == "col" ? col_play_dist : row_play_dist
            opp_play = play_from_int(row.other_choice .+ 1)
            play = play_from_int(row.choice .+ 1)
            (;row..., row_game, col_game, game, row_play_dist, col_play_dist, opp_dist, opp_play, play, self_dist)
        end |> DataFrame
        new_df
    end

    function load_treatment_data(treatment)
        files = glob("data/processed/$treatment/*_play_distributions.csv")
        data = vcat(map(load_data, files)...)
    end


    function load_ind_data(treatment)
        files = glob("data/processed/ind_data/$treatment/*_play_distributions.csv")
        ind_data = vcat(map(load_ind_treat, files)...)
    end
    function fit_ind_mh!(mh_in::MetaHeuristic, games, actual, opp_h, costs; init_x=nothing, loss_f = likelihood)
        mh = deepcopy(mh_in)
        if init_x == nothing
            init_x = [get_parameters(mh)..., mh.prior...]
        end
        function loss_wrap(x)
            set_parameters!(mh, x[1:(end-3)])
            mh.prior = x[end-2:end]
            prediction_loss(mh, games, actual, opp_h, costs)
        end
        # x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS())) # TODO: get autodiff to work with logarithm in likelihood
        x = Optim.minimizer(optimize(loss_wrap, init_x, BFGS(); autodiff = :forward))
        out_mh = deepcopy(mh_in)
        set_parameters!(out_mh, x[1:end-3])
        out_mh.prior = x[end-2:end]
        (opt_mh=out_mh, opt_x=x)
    end

    function fit_model(base_model::MetaHeuristic, games, ind_ch, opp_h, costs::Costs; n_iter=5)
        # games, plays = invert(data)
        # empirical_play = CacheHeuristic(games, plays);
        model = deepcopy(base_model)
        for i in 1:n_iter
            fit_prior!(model, games, ind_ch, opp_h, costs)
            fit_h!(model, games, ind_ch, opp_h, costs)
        end
        model
    end
end

#%%

data.col_game[38].row


pos_data = load_ind_data("positive")
neg_data = load_ind_data("negative")

pos_data.choice .+= 1
neg_data.choice .+= 1
pos_data.other_choice .+= 1
neg_data.other_choice .+= 1

data = vcat(pos_data, neg_data)

ind_ch = CacheHeuristic(vcat(data.game, transpose.(data.game)), vcat(data.play, data.opp_play))
opp_ch = CacheHeuristic(transpose.(data.game), data.opp_dist)
ch = CacheHeuristic([data.row_game..., data.col_game...], [data.row_play_dist..., data.col_play_dist...])

@everywhere ind_ch = deepcopy($ind_ch)
@everywhere opp_ch = deepcopy($opp_ch)
@everywhere ch = deepcopy($ch)

@everywhere begin
    mh_base = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
    C = Costs(0.40944996933250777, 0.29999999999999993, 0.13595487880214152, 2.1179160025079473)
end
#%%

collect(groupby(data, :pid))[1:10]

id_df = collect(groupby(data, :pid))[10]

ind_opts = pmap(collect(groupby(data, :pid))) do id_df
    id = id_df.pid[1]
    treatment = id_df.treatment[1]
    session = id_df.session_code[1]
    games = collect(id_df.game)
    failed = 0
    ind_opt_mh = deepcopy(mh_base)
    ind_opt_x = zeros(10)
    try
        ind_opt_mh, ind_opt_x = fit_ind_mh!(mh_base, games, ind_ch, opp_ch, C)
    catch e
        failed = 1
    end
    ind_perf = prediction_loss(ind_opt_mh, games, ind_ch, opp_ch, C)
    my_names = (:id, :ind_perf, :treatment, :session, :failed, [Symbol("X"*string(i)) for i in 1:length(ind_opt_x)]...)
    NamedTuple{my_names}((id, ind_perf, treatment, session, failed, ind_opt_x...))
end

ind_opts_df = DataFrame(ind_opts)


hs = collect(collect.(zip(ind_opts_df[end-2], ind_opts_df[end-1], ind_opts_df[end])))


ind_opts_df[:ind_perf]


priors = my_softmax.(hs)
mean([p[1] for p in priors])
mean([p[2] for p in priors])
mean([p[3] for p in priors])
CSV.write("results/individual_optx.csv", ind_opts_df)

#%% Out of sample predict with individual estimates
ind_opts_train_test = pmap(collect(groupby(data, :pid))) do id_df
    id = id_df.pid[1]
    treatment = id_df.treatment[1]
    session = id_df.session_code[1]
    games = collect(id_df.game)
    train_games = games[1:30]
    test_games = games[filter(x -> (x in [31, 38, 42, 49]) == false, 31:50)]
    failed = 0
    ind_opt_mh = deepcopy(mh_base)
    ind_opt_x = zeros(10)
    try
        ind_opt_mh, ind_opt_x = fit_ind_mh!(mh_base, train_games, ind_ch, opp_ch, C)
    catch e
        failed = 1
    end
    tot_perf = prediction_loss(ind_opt_mh, games, ind_ch, opp_ch, C)
    train_perf = prediction_loss(ind_opt_mh, train_games, ind_ch, opp_ch, C)
    test_perf = prediction_loss(ind_opt_mh, test_games, ind_ch, opp_ch, C)
    my_names = (:id, :ind_perf, :train_perf, :test_perf, :treatment, :session, :failed, [Symbol("X"*string(i)) for i in 1:length(ind_opt_x)]...)
    NamedTuple{my_names}((id, ind_perf, train_perf, test_perf, treatment, session, failed, ind_opt_x...))
end


ind_opts_train_test_df = DataFrame(ind_opts_train_test)

mean(filter(x -> isnan(x) == false, ind_opts_train_test_df[:test_perf]))

#%%


unique_games = unique([data.row_game..., data.col_game...])
plays = [data.row_play_dist..., data.col_play_dist...]


#
play_distribution(opp_ch, transpose(data.game[1531]))
data.game[92].row
data.role[51]
data[118,:]
data.session_code[1500]
data.treatment[1501]



id = data.pid[160]

id_df = @where(data, :pid .== id)


opt_mh = fit_model(mh_base, unique_games, ch, ch, C)
ind_opt_mh = fit_ind_mh!(mh_base, unique_games, ch, ch, C)
prediction_loss(opt_mh, unique_games, ch, ch, C)
prediction_loss(ind_opt_mh, unique_games, ch, ch, C)

opt_mh = fit_model(mh_base, id_df.game, ind_ch, ch, C)
ind_opt_mh = fit_ind_mh!(mh_base, id_df.game, ind_ch, ch, C)

prediction_loss(opt_mh, id_df.game, ind_ch, ch, C)
prediction_loss(ind_opt_mh, id_df.game, ind_ch, ch, C)


## Look at relative fit of baseline heuristics


function gen_h_fun(h)
    function res_f(g,c)
        return play_distribution(h, g)[c]
    end
    res_f
end

function gen_p_fun(h)
    function payoff_f(g, opp_c)
        if opp_c == 0
            return 0
        else
            p = play_distribution(h, g)
            return payoff =  p'*g.row[:,opp_c]
        end
    end
    payoff_f
end


function gen_exp_p_fun(h, opp_ch)
    function payoff_f(g)
        expected_payoff(h, opp_ch, g)
    end
end


sens= 1.

h_cell = JointMax(sens)
h_maxmax = RowHeuristic(10., sens)
h_maxmean = RowHeuristic(0., sens)
h_maxmin = RowHeuristic(-10., sens)
h_sim = SimHeuristic([RowHeuristic(0., sens), RowHeuristic(0., sens)])


hs = [("cell", h_cell), ("maxmax", h_maxmax), ("maxmean", h_maxmean), ("maxmin", h_maxmin), ("sim", h_sim)]

# for (namn, h) in hs
#     println(namn)
#     data[Symbol("probs_"*namn)] = gen_h_fun(h).(data.game, data.choice)
#     data[Symbol("payoff_"*namn)] = gen_p_fun(h).(data.game, data.choice)
#     gd = groupby(data, :pid)
#     prev(x) = circshift(x, 1)
#     prev_df = combine(gd, Symbol("payoff_"*namn) => prev, keepkeys=true, ungroup=true)
#     data[Symbol("prev_payoff_"*namn)] = prev_df[Symbol("payoff_"*namn*"_prev")]
#     data[data.round .== 1, Symbol("prev_payoff_"*namn)] = 0
#     data[Symbol("prev_payoff_"*namn)] = prev_df[Symbol("payoff_"*namn*"_prev")]
#     gd = groupby(data, :pid)
#     cum_df = combine(gd, Symbol("prev_payoff_"*namn) => cumsum, keepkeys=true, ungroup=true)
#     data[Symbol("cum_payoff_"*namn)] = cum_df[Symbol("prev_payoff_"*namn*"_cumsum")]
#     data[Symbol("cum_payoff_"*namn)] = data[Symbol("cum_payoff_"*namn)]./(data[:round] .- 1)
#     data[data.round .== 1, Symbol("cum_payoff_"*namn)] = 0
#     data[Symbol("exp_payoff_"*namn)] = gen_exp_p_fun(h, opp_ch).(data.game)
# end
#
# data = @transform(data, mean_prev_payoff = (:prev_payoff_cell .+ :prev_payoff_maxmax .+ :prev_payoff_maxmean .+ :prev_payoff_maxmin .+ :prev_payoff_sim)./5)
# data = @transform(data, mean_cum_payoff = (:cum_payoff_cell .+ :cum_payoff_maxmax .+ :cum_payoff_maxmean .+ :cum_payoff_maxmin .+ :cum_payoff_sim)./5)
#
# CSV.write("results/corr_payoff_prob_sens_"*string(Int(sens))*".csv", data)
#


stacked = map(hs) do (namn, h)
    df = deepcopy(data)
    df[:probs] = gen_h_fun(h).(df.game, df.choice)
    # df[:payoff] = gen_u_fun(h, C).(df.game, df.choice)
    df[:payoff] = gen_p_fun(h).(df.game, df.choice)
    df = @transform(groupby(df, :pid), prev_payoff = circshift(:payoff, 1))
    df[df.round .== 1, :prev_payoff] = 0
    df = @transform(groupby(df, :pid), cum_payoff = cumsum(:prev_payoff)./:round)
    df[df.round .== 1, :cum_payoff] = 0
    df[:exp_payoff] = gen_exp_p_fun(h, opp_ch).(df.game)
    df[:heuristic] = namn
    df
end

stacked_df = vcat(stacked...)
CSV.write("results/stacked_corr_payoff_prob_sens_"*string(Int(sens))*".csv", stacked_df)


data[[:round, :payoff_cell, :prev_payoff_cell]]


ps_data = @where(data, :treatment .== "positive")
cor(ps_data[:prev_payoff_cell], ps_data[:probs_cell])
cor(ps_data[:prev_payoff_maxmax], ps_data[:probs_maxmax])
cor(ps_data[:prev_payoff_maxmean], ps_data[:probs_maxmean])
cor(ps_data[:prev_payoff_maxmin], ps_data[:probs_maxmin])
cor(ps_data[:prev_payoff_sim], ps_data[:probs_sim])


gen_p_fun(h_cell).([g,g], [2, 3])




data = @transform(data, h_cell = gen_h_fun(h_cell).(:game, :choice), p_cell=gen_p_fun(h_cell).(:game, :other_choice))
data = @transform(data, cell_payoff=gen_p_fun(h_cell).(:game, :other_choice))

gen_p_fun(h_cell).(data.game, data.other_choice)








c = id_df.choice[1] + 1
g = id_df.game[1]
play_distribution(h_cell, g)[c]
play_distribution(h_maxmax, g)[c]
play_distribution(h_maxmean, g)[c]
play_distribution(h_maxmin, g)[c]
play_distribution(h_sim, g)[c]


#%% Individual learning rule with reinforcement


@everywhere mutable struct RLearning
    mh::MetaHeuristic
    β::Real
    ρ::Real
    costs::Costs
end



@everywhere function rule_loss_ind(rl::RLearning, ind_ch, opp_ch, games)
    pred_loss = 0
    mh = deepcopy(rl.mh)
    for g in games
        for i in 1:length(mh.h_list)
            mh.prior[i] = rl.ρ * mh.prior[i] + rl.β * perf(mh.h_list[i], [g], ind_ch, rl.costs)
        end
        # pred_loss = pred_loss + prediction_loss_no_RI(mh, [g], actual_h, rl.costs) # This is to get sum of loss instead of average
        pred_loss = pred_loss + prediction_loss(mh, [g], ind_ch, opp_ch, rl.costs) # This is to get sum of loss instead of average
    end
    return pred_loss/length(games)
end

function rule_loss(rl, ind_ch, opp_ch, all_games)
    res = map(all_games) do games
        rule_loss_ind(rl, ind_ch, opp_ch, games)
    end
    mean(res)
end


@everywhere function pred_with_rl(rl::RLearning, cums, ind_ch, opp_ch, g)
    mh = deepcopy(rl.mh)
    for i in length(mh.prior)
        mh.prior[i] += rl.β*cums[i]
    end
    prediction_loss(mh, [g], ind_ch, opp_ch, rl.costs)
end

@everywhere begin
    mh_r = MetaHeuristic([JointMax(1.), RowHeuristic(1., 1.), RowHeuristic(0., 1.), RowHeuristic(-1., 1.), SimHeuristic([RowHeuristic(0., 1.), RowHeuristic(0., 1.)])], [0., 0., 0., 0., 0.]);
    C = Costs(0.40944996933250777, 0.29999999999999993, 0.13595487880214152, 2.1179160025079473)
    rl = RLearning(mh_r, 1., 0.95, C)
end


cum_syms = [Symbol("cum_utility_"*namn) for (namn, h) in hs]

rl_pred_dat = map(eachrow(data)) do row
    (g=row.game, cums=collect(row[cum_syms]), treat=row.treatment)
end



play_distribution(opp_ch, transpose(data.game[31]))



@everywhere begin
    out_rl = $out_rl
    out_rl.β = -0.01
end


perfs = pmap(rl_pred_dat) do row
    pred_with_rl(out_rl, row.cums, ind_ch, opp_ch, row.g)
end
mean(perfs)

function opt_fun(x)
    rl.mh.prior = x[1:5]
    rl.β = x[end]
    perfs = map(rl_pred_dat) do row
        pred_with_rl(rl, row.cums, ind_ch, opp_ch, row.g)
    end
    return mean(perfs)
end




init_x = [rl.mh.prior..., 1.]
opt_rl_res = optimize(opt_fun, init_x, BFGS(); autodiff = :forward)
out_rl = deepcopy(rl)
opt_x = Optim.minimizer(opt_rl_res)
out_rl.mh.prior = opt_x[1:5]
out_rl.β = opt_x[end]

set_parameters!(out_mh, x[1:end-3])
out_mh.prior = x[end-2:end]

row_tab = collect(eachrow(data))

row_tab[10][cum_syms]
cums = collect(data[10, cum_syms])

pred_with_rl(rl, cums, ind_ch, opp_ch, data.game[10])



function rule_loss(rl:RLearning, data::DataFrame, ind_ch, opp_ch, games)






per_ind_pos_games = map(collect(groupby(pos_data, :pid))) do id_df
    collect(id_df.game)
end

per_ind_neg_games = map(collect(groupby(neg_data, :pid))) do id_df
    collect(id_df.game)
end

rule_loss(rl, ind_ch, opp_ch, per_ind_neg_games)
@time rule_loss(rl, ind_ch, opp_ch, per_ind_pos_games)

rule_loss_ind(rl, ind_ch, opp_ch, per_ind_neg_games[1])



function gen_u_fun(mh::MetaHeuristic, C::Costs, opp_dist)
    function payoff_f(g, opp_c)
        if opp_c == 0
            return 0
        else
            p = play_distribution(mh, g, opp_dist, C)
            cost = sum(h_distribution(mh, g, opp_dist, C) .* C.(mh.h_list))
            return payoff =  p'*g.row[:,opp_c] - cost
        end
    end
    payoff_f
end

play_distribution(opp_ch, transpose(data.game[1]))

function gen_u_fun(h::Heuristic, C::Costs)
    function payoff_f(g, opp_c)
        if opp_c == 0
            return 0
        else
            p = play_distribution(h, g)
            return payoff =  p'*g.row[:,opp_c] - C(h)
        end
    end
    payoff_f
end


for (namn, h) in hs
    u_fun = gen_u_fun(h, C)
    u_sym = Symbol("utility_"*namn)
    data[u_sym] = u_fun.(data.game, data.other_choice)
    cum_u_sym = Symbol("cum_utility_"*namn)
    gd = groupby(data, :pid)
    prev(x) = circshift(x,1)
    cum_df = combine(groupby(data, :pid), u_sym => cumsum, keepkeys=true, ungroup=true)
    cum_df = combine(groupby(cum_df, :pid), Symbol(string(u_sym)*"_cumsum") => prev, keepkeys=true, ungroup=true)
    data[cum_u_sym] = cum_df[Symbol(string(u_sym)*"_cumsum_prev")]
    data[data.round .== 1, cum_u_sym] = 0
end





pos_data.type
dat = @where(neg_data, :type .== "treatment")

h = h_sim

res_mean =  map(zip(dat.game, dat.choice, dat.other_choice)) do (g, c, opp_c)
    o_c = opp_c < 1 ?  1 : opp_c
    # mean(g.row[:,o_c])
    p = play_distribution(h, g)
    mean(p'*g.row[:,o_c])
    # mean(g.row[c,o_c])
end

mean(res_mean)

for (namn, h) in hs
    u_fun = gen_u_fun(h, C)
    u_sym = Symbol("utility_"*namn)
    data[u_sym] = u_fun.(data.game, data.other_choice)
end




u_fun = gen_u_fun(mh_r, C, opp_ch)

u_fun_h = gen_u_fun(mh_r.h_list[1], C)

u_fun_h(data.game[1], 2)
