include("Heuristics.jl")


mutable struct RuleLearning
    mh::MetaHeuristic
    β₀::Float64
    β₁::Float64
    costs::Costs
end

function gen_data_dict(data)
    data_dicts = Dict()
    for n in 1:Int(length(data)/100)
        start_idx = (n-1)*100
        row_start = start_idx + 1
        row_end = start_idx + 50
        col_start = start_idx + 51
        col_end = start_idx + 100
        dict_to_add = Dict("row" => [Dict("game"=> data[i][1], "play"=> data[i][2]) for i in row_start:row_end],
                      "col" => [Dict("game"=> data[i][1], "play"=> data[i][2]) for i in col_start:col_end])
        data_dicts[n] = dict_to_add
    end
    return data_dicts
end

function gen_opp_h_dict(data_dict)
    opp_h_dict = Dict()
    for (key, val_dict) in data_dict
        opp_h_dict[key] = Dict()
        opp_h_dict[key]["row"] = CacheHeuristic([x["game"] for x in val_dict["row"]], [x["play"] for x in val_dict["row"]])
        opp_h_dict[key]["col"] = CacheHeuristic([x["game"] for x in val_dict["col"]], [x["play"] for x in val_dict["col"]])
    end
    return opp_h_dict
end


function rule_loss(rl::RuleLearning, data)
    data_dict = gen_data_dict(data)
    opp_h_dict = gen_opp_h_dict(data_dict)
    pred_loss = 0
    for treat in keys(data_dict), role in ["row", "col"]
        mh = deepcopy(rl.mh)
        opp_role = role == "row" ? "col" : "row"
        opp_h = opp_h_dict[treat][opp_role]
        actual_h = opp_h_dict[treat][role]
        actual_h = opp_h_dict[treat][role]
        for r in 1:50
            game = data_dict[treat][role][r]["game"]
            for i in 1:length(mh.h_list)
                mh.prior[i] = rl.β₀ * mh.prior[i] + rl.β₁ * perf(mh.h_list[i], [game], opp_h, rl.costs)
            end
            # pred_loss = pred_loss + prediction_loss_no_RI(mh, [game], actual_h, rl.costs) # This is to get sum of loss instead of average
            pred_loss = pred_loss + prediction_loss(mh, [game], actual_h, opp_h, rl.costs) # This is to get sum of loss instead of average
        end
    end
    return pred_loss/length(data)
end

function rule_loss_idx(rl::RuleLearning, data, idx)
    data_dict = gen_data_dict(data)
    opp_h_dict = gen_opp_h_dict(data_dict)
    pred_loss = 0
    l_num = 0
    for treat in keys(data_dict), role in ["row", "col"]
        mh = deepcopy(rl.mh)
        loss_idx = filter(x -> (x > (treat - 1)*100 && x <= (treat -1)*100 + 50), idx)
        opp_role = role == "row" ? "col" : "row"
        opp_h = opp_h_dict[treat][opp_role]
        actual_h = opp_h_dict[treat][role]
        actual_h = opp_h_dict[treat][role]
        for r in 1:50
            game = data_dict[treat][role][r]["game"]
            for i in 1:length(mh.h_list)
                mh.prior[i] = rl.β₀ * mh.prior[i] + rl.β₁ * perf(mh.h_list[i], [game], opp_h, rl.costs)
            end
            # pred_loss = pred_loss + prediction_loss_no_RI(mh, [game], actual_h, rl.costs) # This is to get sum of loss instead of average
            if r in loss_idx
                pred_loss = pred_loss + prediction_loss(mh, [game], actual_h, opp_h, rl.costs) # This is to get sum of loss instead of average
                l_num += 1
            end
        end
    end
    return pred_loss/l_num
end



function end_rules(rl::RuleLearning, data)
    data_dict = gen_data_dict(data)
    opp_h_dict = gen_opp_h_dict(data_dict)
    treats = keys(data_dict)
    rules = Dict(t => Dict() for t in treats)
    for treat in treats, role in ["row", "col"]
        mh = deepcopy(rl.mh)
        opp_role = role == "row" ? "col" : "row"
        opp_h = opp_h_dict[treat][opp_role]
        actual_h = opp_h_dict[treat][role]
        for r in 1:50
            game = data_dict[treat][role][r]["game"]
            for i in 1:length(mh.h_list)
                mh.prior[i] = rl.β₀ * mh.prior[i] + rl.β₁ * perf(mh.h_list[i], [game], opp_h, rl.costs)
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

function fit_βs_and_prior(rl_base::RuleLearning, data, idx)
    data_dict = gen_data_dict(data)
    opp_h_dict = gen_opp_h_dict(data_dict)
    rl = deepcopy(rl_base)
    init_x = [rl.β₀, rl.β₁, rl.mh.prior...]
    function loss_f(x)
        rl.β₀ = x[1]
        rl.β₁ = x[2]
        rl.mh.prior = x[3:end]
        return rule_loss_idx(rl, data, idx)
    end
    res_x = Optim.minimizer(optimize(loss_f, init_x))
    rl.β₀ = res_x[1]
    rl.β₁ = res_x[2]
    rl.mh.prior = res_x[3:end]
    return rl
end

function optimize_rule_lambdas(rl_base::RuleLearning, data, idx)
    data_dict = gen_data_dict(data)
    opp_h_dict = gen_opp_h_dict(data_dict)
    rl = deepcopy(rl_base)
    init_x = get_lambdas(rl.mh)
    function loss_f(x)
        set_lambdas!(rl.mh, x)
        return rule_loss_idx(rl, data, idx)
    end
    res_x = Optim.minimizer(optimize(loss_f, init_x))
    set_lambdas!(rl.mh, res_x)
    return rl
end
