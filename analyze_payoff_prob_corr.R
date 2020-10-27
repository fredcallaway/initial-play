library(stargazer)
library(Hmisc)
library(dplyr)
library(tidyr)
library(ggplot2)
library(xtable)
library(stargazer)
library(caret)
library(lfe)
library(randomForest)
library(lightgbm)
library(xgboost)
library(forcats)
library(mltools)
library(reshape2)
library(data.table)

#
df = read.csv("results/stacked_corr_payoff_prob_sens_20.csv")
df$heuristic
df$cum_payoff
df$prev_payoff

df$probs_cell
df$pid

df$probs

model = felm(probs ~ cum_payoff  +  exp_payoff | treatment*heuristic*factor(round), data=df)
summary(model)

model = felm(probs ~ cum_payoff + exp_payoff | session_code*heuristic*factor(round), data=df)
summary(model)


summary(df$payoff)


group_by(df[df$round > 30,], treatment, heuristic) %>% summarise(men_p = mean(payoff))


model = felm(probs ~ cum_payoff + exp_payoff | treatment*heuristic, data=df)
summary(model)



