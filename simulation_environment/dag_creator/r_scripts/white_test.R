library(lavaan)
data(PoliticalDemocracy)
PoliticalDemocracy
write.csv(PoliticalDemocracy, "Thesis/code/thesis/simulation_environment/dag_creator/real_data/politicaldemocracy.csv", row.names = FALSE)
plot(density(PoliticalDemocracy$x2))
pol_data = read.csv("real_data/poldem_values.csv")

library(tseries)
library(gtools)

setwd("C:/Users/JB/Documents/Thesis/Code/thesis/simulation_environment/dag_creator")

spirtes_sim = read.csv("simulated_data/spirtes_nonlin_random_b0.1_d0.25_samples1000_n3.csv")

comb_list = combinations(length(colnames(spirtes_sim)), 2, colnames(spirtes_sim))
median_list = c()


for (num in 0:99) {
  string_name = sprintf("simulated_data/spirtes_nonlin_random_b0.1_d0.25_samples1000_n%s.csv", num)
  spirtes_sim = read.csv(string_name)
  test_list = c()
  for (index in 1:(dim(comb_list)[1])) {
    test_list[index] <- white.test(spirtes_sim[comb_list[index,1]], spirtes_sim[comb_list[index,2]])$p.value
  }
  median_list[num] <- median(sort(test_list))
}

mean(median_list)
#test_list
