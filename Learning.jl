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
using ForwardDiff
using ReverseDiff
using BlackBoxOptim



nprocs() == 1 && addprocs(19)
# Outline
# 1. Calc derivative of params with respect to payoff for a single game.
# 2.
#%%

include("Heuristics.jl")

@everywhere begin
    using ForwardDiff
end


@everywhere begin
    include("Heuristics.jl")
    include("model.jl")
    include("box.jl")
end

#%%

all_data = Dict(
    :pos => load_treatment_data("positive"),
    :neg => load_treatment_data("negative"),
)


train_idx, test_idx = early_late_indices(vcat(all_data[:pos]))


games_pos, plays_pos = invert(all_data[:pos])
games_neg, plays_neg = invert(all_data[:neg])

games = [games_pos..., games_neg...]
plays = [plays_pos..., plays_neg...]

ch = CacheHeuristic(games, plays);

games_per_sess = [(games[1 + (i*100):(i)*100+50],games[51 + (i*100):(i)*100+100])  for i in 0:19]
plays_per_sess = [(plays[1 + (i*100):(i)*100+50],plays[51 + (i*100):(i)*100+100])  for i in 0:19]

test_games_per_sess = [(games[31 + (i*100):(i)*100+50],games[81 + (i*100):(i)*100+100])  for i in 0:19]
train_games_per_sess = [(games[1 + (i*100):(i)*100+30],games[51 + (i*100):(i)*100+80])  for i in 0:19]

row_plays = vcat([plays[1 + (i*100):(i)*100+50]  for i in 0:19]...)
col_plays = vcat([plays[51 + (i*100):(i)*100+100]  for i in 0:19]...)

@everywhere begin
    games = $games
    plays = $plays
    games_per_sess = $games_per_sess
    plays_per_sess = $plays_per_sess
    train_games_per_sess = $train_games_per_sess
    row_plays = $row_plays
    col_plays = $col_plays
    train_row_plays = $row_plays[$train_idx]
    train_col_plays = $col_plays[$train_idx]
end


#%%
@everywhere begin
    mutable struct Learning
        mh::MetaHeuristic
        C::Costs
        λ::Real
    end

    get_parameters(c::Costs) = [C.α, c.λ, c.level, c.m_λ]

    function get_parameters(l::Learning)
        # return [get_parameters(l.mh)..., l.mh.prior..., get_parameters(l.C)..., l.λ]
        return [get_parameters(l.mh)..., l.mh.prior..., l.λ]
    end


    function set_parameters!(l::Learning, x)
        idx = 1
        idx_new = length(get_parameters(l.mh))
        set_parameters!(l.mh, x[idx:idx_new])
        idx = idx_new +1
        idx_new = idx + length(l.mh.prior) -1
        l.mh.prior = x[idx:idx_new]
        # l.C = Costs(x[idx_new+1:end-1]...)
        l.λ = x[end]
        l
    end

    function get_params_for_learn(mh::MetaHeuristic)
        return [get_parameters(mh)..., mh.prior...]
    end

    function set_params_for_learn(mh_in::MetaHeuristic, x)
        mh = deepcopy(mh_in)
        set_parameters!(mh, x[1:end-3])
        mh.prior = x[end-2:end]
        mh
    end

    function set_params_for_learn!(mh::MetaHeuristic, x)
        set_parameters!(mh, x[1:end-3])
        mh.prior = x[end-2:end]
        mh
    end

    function f_to_derive(x, mh_in, g, ch, C)
        mh = set_params_for_learn(mh_in, x)
        payoff = expected_payoff(mh, ch, g, C)
        payoff - sum(h_distribution(mh, g, ch, C) .* C.(mh.h_list))
    end

    function update_l!(l::Learning, gs::Vector{Game}, ch::CacheHeuristic)
        x = get_params_for_learn(l.mh)
        mh_in = deepcopy(l.mh)
        Δxs = mean([ForwardDiff.gradient(ps -> f_to_derive(ps, mh_in, g, ch, l.C), x) for g in gs])
        x += l.λ*(Δxs)./2
        set_params_for_learn!(l.mh, x)
    end

    function pred_with_learning(l_in::Learning, games_per_sess, ch)
        all_preds = pmap(games_per_sess) do games
            l = deepcopy(l_in)
            preds = map(zip(games[1], games[2])) do (g_row, g_col)
                pred_row = play_distribution(l.mh, g_row, ch, l.C)
                pred_col = play_distribution(l.mh, g_col, ch, l.C)
                update_l!(l, [g_row, g_col], ch)
                (pred_row, pred_col)
            end
            preds
        end
        vcat(all_preds...)
    end

    mh_base  = MetaHeuristic([JointMax(3.), RowHeuristic(0., 2.), SimHeuristic([RowHeuristic(1., 1.), RowHeuristic(0., 2.)])], [0., 0., 0.]);
    C = Costs(0.40944996933250777, 0.29999999999999993, 0.13595487880214152, 2.1179160025079473)
    l_in = Learning(mh_base, C, 1.)

    function wrap_f(x)
        l = set_parameters!(l_in, x)
        preds = pred_with_learning(l, games_per_sess, ch)
        mean([likelihood.([p[1] for p in preds], row_plays)..., likelihood.([p[2] for p in preds], col_plays)...])
    end

    function train_wrap_f(x)
        l = set_parameters!(l_in, x)
        preds = pred_with_learning(l, train_games_per_sess, ch)
        mean([likelihood.([p[1] for p in preds], train_row_plays)..., likelihood.([p[2] for p in preds], train_col_plays)...])
    end
end


#%% Testing forward diff
l_in = Learning(mh_base, C, 1.)
x = get_parameters(l_in)

#%%
# srange = [(0.,5.), (-5., 5.), (0.,5.), (-5., 5.), (0., 5.), (-5., 5.), (0., 5.), (0. ,5.), (0., 5.), (0., 5.), (0., 5.),(0., 5.),(0., 5.),(0., 5.), (0., 10.)]
srange = [(0.,5.), (-5., 5.), (0.,5.), (-5., 5.), (0., 5.), (-5., 5.), (0., 5.), (0. ,5.), (0., 5.), (0., 5.), (0., 10.)]
res = bboptimize(wrap_f, SearchRange = srange, NumDimensions=length(srange), MaxTime=360, TraceInterval=100, TraceTraceMode=:compact)
opt_x = best_candidate(res)
open("saved_objects/fixC_full_learning_bb_opt.json", "w") do f
    write(f, JSON.json(opt_x))
end
opt_in = Optim.optimize(wrap_f, opt_x, NelderMead(), Optim.Options(time_limit=180, show_trace=true))
nm_opt = Optim.minimizer(opt_in)
open("saved_objects/fixC_full_learning_nm_opt.json", "w") do f
    write(f, JSON.json(nm_opt))
end

res_train = bboptimize(train_wrap_f, SearchRange = srange, NumDimensions=length(srange), TraceInterval=10, MaxTime=360, TraceMode=:compact)
opt_x_train = best_candidate(res_train)
open("saved_objects/fixC_train_learning_bb_opt.json", "w") do f
    write(f, JSON.json(opt_x_train))
end

opt_in_NM_train = Optim.optimize(train_wrap_f, opt_x_train, NelderMead(), Optim.Options(time_limit=360, show_trace=true))
opt_in_LBFGS_train = Optim.optimize(train_wrap_f, opt_x_train, LBFGS(), Optim.Options(time_limit=360, show_trace=true), autodiff = :forward)
nm_opt_train = Optim.minimizer(opt_in_NM_train)
open("saved_objects/fixC_train_learning_nm_opt.json", "w") do f
    write(f, JSON.json(nm_opt_train))
end

l = set_parameters!(deepcopy(l_in), nm_opt)
l = set_parameters!(deepcopy(l_in), nm_opt_train)

preds = pred_with_learning(l, games_per_sess, ch)

pos_test_idx = filter(x -> x <= 500, test_idx)
neg_test_idx = filter(x -> x > 500, test_idx)

row_preds = [p[1] for p in preds]
col_preds = [p[2] for p in preds]
test_preds_row = row_preds[test_idx]
test_preds_col = col_preds[test_idx]

pos_test_row = mean(likelihood.(row_preds[pos_test_idx], row_plays[pos_test_idx]))
neg_test_row = mean(likelihood.(row_preds[neg_test_idx], row_plays[neg_test_idx]))

pos_test_col = mean(likelihood.(col_preds[pos_test_idx], col_plays[pos_test_idx]))
neg_test_col = mean(likelihood.(col_preds[neg_test_idx], col_plays[neg_test_idx]))

mean([pos_test_row, pos_test_col])
mean([neg_test_row, neg_test_col])

l.mh.h_list
l.mh.prior

l.C

games = games_per_sess[5]

preds = map(zip(games[1], games[2])) do (g_row, g_col)
    pred_row = play_distribution(l.mh, g_row, ch, l.C)
    pred_col = play_distribution(l.mh, g_col, ch, l.C)
    update_l!(l, [g_row, g_col], ch)
    (pred_row, pred_col)
end

l.mh.h_list
