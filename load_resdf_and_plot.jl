using Plots
using StatsPlots
using DataFrames
using CSV
pyplot()

res_df = CSV.read("res_df_from_pilot.csv")



data_names = [:random, :neg_QCH, :pos_QCH, :opt_mh_neg, :opt_mh_pos, :fit_mh_neg, :fit_mh_pos, :opt_deep_neg, :opt_deep_pos, :fit_deep_neg,  :fit_deep_pos, :minimum]
plots_vec = []
for data_type in ["all", "train", "test", "comparison"], treat in ["negative", "positive"]
    vals = convert(Vector, first(res_df[(res_df.treatment .== treat) .& (res_df.data_type .== data_type), data_names]))
    ctg = [repeat(["minimum"], 5)..., repeat(["negative"], 5)..., repeat(["positive"], 5)..., repeat(["random"], 5)...]
    nam = [repeat(["QCH", "fit mh", "opt mh", "fit deep", "opt deep"],4)...]
    bar_vals = hcat(repeat([vals[12]],5), transpose(reshape(vals[2:(end-1)], (2,5))), repeat([vals[1]],5))
    bar_vals = convert(Array{Float64,2}, bar_vals)
    push!(plots_vec, groupedbar(nam, bar_vals, group=ctg, lw=0, framestyle = :box, title = treat*"-"*data_type))
end

length(plots_vec)
plot(plots_vec..., layout=(4,2), size=(1191,1684))

savefig("test.png")

res_df
