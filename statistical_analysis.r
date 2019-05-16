library(jsonlite)
pos_data = read.csv("data/processed/positive/pilot_pos_play_distributions.csv", stringsAsFactors = FALSE)
neg_data = read.csv("data/processed/negative/pilot_neg_play_distributions.csv", stringsAsFactors = FALSE)

pos_data

[31, 34, 38, 41, 44, 50, 131, 137, 141, 144, 149]

idx = 33
pos_play = round((fromJSON(pos_data[idx, "row_play"]) + fromJSON(pos_data[idx, "col_play"]))/2 * 30)
neg_play = round((fromJSON(neg_data[idx, "row_play"]) + fromJSON(neg_data[idx, "col_play"]))/2 * 30)
pos_play
neg_play
data = matrix(c(pos_play, neg_play), nrow=2)

result <- chisq.test(data)
result
