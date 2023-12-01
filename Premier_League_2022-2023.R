##########################################################
# Load required packages
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "https://cran.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "https://cran.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "https://cran.r-project.org")
if(!require(caret)) install.packages("caret", repos = "https://cran.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = c("https://cran.rediris.org/", "https://cloud.r-project.org/"))
if(!require(randomForest)) install.packages("randomForest", repos = "https://cran.r-project.org")

library(tidyverse)
library(dplyr)
library(purrr)
library(caret)
library(ggpubr)
library(randomForest)

##########################################################
# Data preparation
##########################################################

# Load CSV file and store the data in a data frame
# Premier_League.csv can be downloaded from 
# https://www.kaggle.com/datasets/thamersekhri/premier-league-stats-2022-2023/
stats_file <- "Premier_League.csv"
premier_league_stats <- read.csv(stats_file)

# Explore premier_league_stats data structure
str(premier_league_stats)

# Rename columns and converting data types for data further exploration and processing
premier_league_stats <- premier_league_stats %>% rename("home_team"="Home.Team",
                                                        "away_team"="Away.Team",
                                                        "home_goals"="Goals.Home",
                                                        "away_goals"="Away.Goals",
                                                        "home_on_target"="home_on",
                                                        "away_on_target"="away_on",
                                                        "home_off_target"="home_off",
                                                        "away_off_target"="away_off"
)

#Convert home team name from char to factor
premier_league_stats$home_team <- as.factor(premier_league_stats$home_team) 

#Convert away team name from char to factor
premier_league_stats$away_team <- as.factor(premier_league_stats$away_team) 

#Convert attendance from char to numeric
premier_league_stats$attendance <- as.numeric(gsub(',','',premier_league_stats$attendance)) 

#Find a column containing NA
colSums(is.na(premier_league_stats))

# Create a new column named "result" to summarize the match result to H/D/A
# H means home team wins, D means draw and A means away team wins
premier_league_stats$result <- apply(premier_league_stats[, c('home_goals', 'away_goals')], 1, 
                                     function(row) {if (row[1] > row[2]) {
                                       return("H")
                                     } else if (row[1] == row[2]) {
                                       return("D")
                                     } else {
                                       return("A")
                                     }
                                     })

# Re-order levels in result column
premier_league_stats$result <- factor(premier_league_stats$result, levels=c('H', 'D', 'A'))

# Delete unused columns
premier_league_stats <- subset(premier_league_stats, 
                               select = -c(date, clock, stadium, attendance, 
                                           home_goals, away_goals, links))

# Display data frame structure
str(premier_league_stats)

# Check if we have NA in the data set
sum(is.na(premier_league_stats))

##########################################################
# Create test and train data sets
##########################################################
set.seed(1)
test_index <- createDataPartition(premier_league_stats$result, times = 1, 
                                  p = 0.3, list = FALSE)
test_set <- premier_league_stats[test_index,]
train_set <- premier_league_stats[-test_index,]

# Check the size of test set and train set
nrow(test_set)
nrow(train_set)

##########################################################
# Find probability of randomly guessing a result from 3-class classification
##########################################################
set.seed(1)
results <- replicate(10000, {
  guess = sample(c('H', 'D', 'A'), nrow(test_set), replace=TRUE)
  mean(guess==test_set$result)
})
mean(results)

##########################################################
# Model 1: Home team effect
##########################################################
# Plot to virtualize home team effect
ggplot(train_set, aes(x=result)) + 
  geom_bar(stat="count", width=0.7, fill="steelblue")

# Predict all home teams to win
model_1_accu <- mean(test_set$result=='H')

# Save result to a data frame for further display
results <- data_frame(Method = "Model 1: Home team effect", 
                      Accuracy = round(model_1_accu, 7))

results %>% knitr::kable()

##########################################################
# Data exploration: Find predictors via Kruskal-Wallis Test
##########################################################

# Put all column names into a variable, except result
c_names <- colnames(train_set)
c_names <- c_names[ !c_names == 'result']

# Run Kruskal-Wallis test
kruskal_test<-sapply(c_names , function(c){
  p<-kruskal.test(train_set$result ~ train_set[,c])$p.value
  return(p)
})

# Put Kruskal-Wallis test results in data frame format and also check if any variables is significant
kruskal_test_df <- data.frame(predictors = names(kruskal_test), 
                              p_value = kruskal_test, row.names = NULL)
kruskal_test_df$significance <- kruskal_test_df$p_value < 0.05
kruskal_test_df %>%  arrange(p_value)


##########################################################
# Data exploration: Find predictors via data observation (box plots)
##########################################################

# Create box plots for all variables except home_team & away_team 
# since they are not suitable for box plots.
boxplot_list <-lapply(c_names[3:32], function(column) {
  train_set %>% ggplot(aes(result, !!sym(column), fill = result)) +
    geom_boxplot(alpha = 0.2) + theme(axis.title.x = element_blank()) 
})

# Put box plots into a figure and display them
figure1 <- ggarrange(plotlist = boxplot_list[1:15], ncol = 5, nrow = 3,
                     common.legend = TRUE, legend = "bottom")
figure1

figure2 <- ggarrange(plotlist = boxplot_list[16:30], ncol = 5, nrow = 3,
                     common.legend = TRUE, legend = "bottom")
figure2

##########################################################
# Model 2: Model 2: Rpart with predictor list from box plot observation
##########################################################

train_rpart_b <- train( result ~ home_shots 
                        + away_shots + home_on_target + away_on_target + away_pass
                        + home_chances + away_chances + home_offside + away_offside,
                        method = "rpart",
                        data = train_set)

model_2_accu <- confusionMatrix(predict(train_rpart_b, test_set, type = "raw"), 
                                test_set$result)$overall["Accuracy"]

results <- rbind(results,c("Model 2: Rpart with predictor list from box plot observation", round(model_2_accu, 7)))
results %>% knitr::kable()

##########################################################
# Model 3: Model 3: Rpart with predictor list from Kruskal-Wallis Test
##########################################################

train_rpart_k <- train( result ~ home_team + away_team + away_shots + home_on_target 
                        + away_on_target + home_chances + away_chances + home_red,
                        method = "rpart",
                        data = train_set)

model_3_accu <- confusionMatrix(predict(train_rpart_k, test_set, type = "raw"), 
                                test_set$result)$overall["Accuracy"]

results <- rbind(results,c("Model 3: Rpart with predictor list from Kruskal-Wallis Test", round(model_3_accu, 7)))
results %>% knitr::kable()

#Compare model 2 prediction to model 3 prediction
identical(predict(train_rpart_b, test_set, type = "raw"), 
          predict(train_rpart_k, test_set, type = "raw"))

##########################################################
# Model 4: KNN with predictor list from box plot observation 
##########################################################

set.seed(1)
train_knn_b <- train(result ~ home_shots 
                     + away_shots + home_on_target + away_on_target + away_pass
                     + home_chances + away_chances + home_offside + away_offside, 
                     tuneGrid = expand.grid(k = seq(50, 100, by = 1)), 
                     method ="knn", 
                     data = train_set)

model_4_accu <- confusionMatrix(predict(train_knn_b, test_set, type = "raw"), 
                                test_set$result)$overall[["Accuracy"]]

results <- rbind(results,c("Model 4: KNN with predictor list from box plot observation", round(model_4_accu, 7)))
results %>% knitr::kable()

##########################################################
# Model 5: KNN with predictor list from Kruskal-Wallis Test
##########################################################
set.seed(1)
train_knn_k <- train(result ~ home_team + away_team + away_shots + home_on_target + 
                       away_on_target + home_chances + away_chances + home_red, 
                     tuneGrid = expand.grid(k = seq(50, 100, by = 1)), 
                     method ="knn", 
                     data = train_set)

model_5_accu <- confusionMatrix(predict(train_knn_k, test_set, type = "raw"), 
                                test_set$result)$overall[["Accuracy"]]

results <- rbind(results,c("Model 5: KNN with predictor list from Kruskal-Wallis Test", round(model_5_accu, 7)))
results %>% knitr::kable()

##########################################################
# Model 6: Random Forest with all predictors
##########################################################
set.seed(1)
train_rf_a <- randomForest(result ~ ., data=train_set,na.action = na.pass)

model_6_accu <-confusionMatrix(predict(train_rf_a, test_set), test_set$result)$overall["Accuracy"]

results <- rbind(results,c("Model 6: Random Forest with all predictors", round(model_6_accu, 7)))
results %>% knitr::kable()


##########################################################
# Model 7: Random Forest with predictor list from box plot observation
##########################################################
set.seed(1)
train_rf_b <- randomForest(result ~home_shots 
                           + away_shots + home_on_target + away_on_target + away_pass
                           + home_chances + away_chances + home_offside + away_offside,
                           data=train_set,na.action = na.pass)

model_7_accu <- confusionMatrix(predict(train_rf_b, test_set), test_set$result)$overall["Accuracy"]

results <- rbind(results,c("Model 7: Random Forest with predictor list from box plot observation", round(model_7_accu, 7)))
results %>% knitr::kable()

##########################################################
# Model 8: Model 8: Random Forest with predictor list from Kruskal-Wallis Test
##########################################################
set.seed(1)
train_rf_k <- randomForest(result ~ home_team + away_team + away_shots + home_on_target 
                           + away_on_target + home_chances + away_chances + home_red,
                           data=train_set,na.action = na.pass)
model_8_accu <- confusionMatrix(predict(train_rf_k, test_set), test_set$result)$overall["Accuracy"]

results <- rbind(results,c("Model 8: Random Forest with predictor list from Kruskal-Wallis Test", round(model_8_accu, 7)))
results %>% knitr::kable()