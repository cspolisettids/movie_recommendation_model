# title: "Movie Recommendation Model"
# author: "Chandra Sekhar Polisetti"
# date: "12/03/2021"
#############################################################################################################
#                             Program Layout ----
#   This program is made of the below steps, 
#   there is a detailed description of the code at the beginning of each step
#   Step0: Install and load the packages required to run the program
#   Step1: Download Capstone Project dataset and clean the data. This gets us movielense dataset
#   Step2: Data Partition for Validation and Training. movielense is made into edx and validation datasets
#   Step3: Data Wrangling to add Movie Release Year and Movie Rated Date
#   Step4: Algorithm evaluation criteria - Define RMSE function and Adjusted R Square Functions
#   Step5: Movie Recommendation Model - Optimization using Cross Validation
#   Step6: Movie Recommendation Model - Validation & Display Results
#############################################################################################################

############################################################################################################
#                          Step0:  Install and load the packages required to run the program ----
###############################################################################################################
#                          Step0:  Brief Summary Of this Step
###############################################################################################################
#    All the required packages to run the program are installed and loaded into the memory
###############################################################################################################
###############################################################################################################
#                         Step0:  R Code
###############################################################################################################

# Install Required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")

# Open required package libraries

library(tidyverse)
library(ggplot2)
library(lubridate)
library(stringr)
library(kableExtra)
library(caret)
library(knitr)
library(scales)
library(ggthemes)
library(glue)

##############################################################################################################
#                          Step1:  Download and Data Cleanup ----
###############################################################################################################
#                          Step1:  Breif Summary Of this Step
###############################################################################################################
# The following are the sequence of steps that are performed for the data cleanup.
# 
# 1) Download the dataset from http://files.grouplens.org/datasets/movielens/ml-10m.zip
# 2) Read the ratings data from "ml-10M100K/ratings.dat" file and name the columns of data frame as "userId", 
#    "movieId", "rating", "timestamp"
# 3) Read the movies data from "ml-10M100K/movies.dat" file and name the columns of data frame as "movieId", 
#    "title", "genres". Convert the movieId to numeric, convert the title and genres to character data types.
# 4) Join the movies and ratings data frames by movieId and call it as movielense.
###############################################################################################################
#                         Step1:  R Code - This code was provided in the capstone project
###############################################################################################################

# Download the dataset from http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Read the ratings data from "ml-10M100K/ratings.dat" 
#file and name the columns of data frame as "userId", "movieId", "rating", "timestamp"

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Read the movies data from "ml-10M100K/movies.dat" file and name the columns of data frame as "movieId", 
# "title", "genres". Convert the movieId to numeric, convert the title and genres to character data types.

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Join the movies and ratings data frames by movieId and call it as movielense.

movielens <- left_join(ratings, movies, by = "movieId")

###############################################################################################################
#                          Step2:  Data Partition for Validation and Training ----
###############################################################################################################
#                          Step2:  Breif Summary Of this Step
###############################################################################################################
# Once clean dataset is obtained in the above step, data in the movielense dataset is split into two parts, 
# one for the training and optimizing the model, and the other for validating the model. 
# 
# Following are the steps that are performed for data partitioning 
# 
# 1) 10% of the data rows from the movielense dataset are randomly selected and placed in validation  dataset, 
#    and this dataset will be kept aside for performing the validation of the final Movie Recommentation Model. 
#    This dataset will not be used for training and optimizing the model. 
#    This dataset is not used for training mainly to avoid overfitting the data.
# 2) The remaining 90% of the data rows from the movielense dataset are brought into the edx dataset.
#    This dataset is mainly used for training and optimizing the algorithms.
#    The edx dataset would be further partitioned into training and test datasets 
#    down the line in the methods section to facilitate the building and optimizing 
#    the Movie Recommendation Model.
# 3) It was made sure that all the movieIds and userIds in validation set are present in the edx dataset. 
#    The movieIds and userIds that are not present in the edx are added back to the edx dataset.
#    This step is performed to make sure that the machine learning algorithms are using the same movies ,
#    and users used during training to make the predictions.
###############################################################################################################
#                         Step2:  R Code - This code was provided by the capstone project
###############################################################################################################

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
# semi join will join temp with edx and only return the rows which have mating movieid and userid; 
# and in this process the records which does not have mating movieid and userid will be filtered out.
# and those filtered records will be added back to edx dataset in the next step

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################################################################################################
#                          Step3:  Data Wrangling to add Movie Release Year and Movie Rated Date----
###############################################################################################################
#                          Step3:  Breif Summary Of this Step
###############################################################################################################
# We need to perform the below tasks, pre processing steps, to make the edx and validation datasets ready
# for training and validation repectively.
# 
# 1) Movie Release Year - We need to incorporate Movie Release Year effect to the Movie Recommendation Model,
#    and for that,we need to add the Movie Release year to the both edx and validation datasets. 
#    Movie Release Year is embedded in the title column, and we need to pull out year from the title and 
#    save it as a separate attribute in the both edx and validation datasets.
# 
# 2) Movie Rated Date - We need to incorporate Movie Rated Date effect to the Movie Recommendation Model, 
#    and Movie Rated Date is not ready available to use it. But we can extract it from the timestamp 
#    field and add it to the edx and validation datasets for further processing. 
#    Firstly Movie Rated Date is obtained by converting timestamp attribute to datetime, 
#    and secondly the Movie Rated Date was rounded to the nearest $week$ for smoothing the data. 
###############################################################################################################
#                         Step3:  R Code
###############################################################################################################

# Add Movie Release Year to edx dataset
edx$year <- as.numeric(substr(as.character(edx$title),
                             nchar(as.character(edx$title))-4,
                             nchar(as.character(edx$title))-1))
# Add Movie Release Year to validation dataset

validation$year <- as.numeric(substr(as.character(validation$title),
                              nchar(as.character(validation$title))-4,
                              nchar(as.character(validation$title))-1))

# Extract Rated date from the timestamp and round it to the nearest week
edx <- edx %>% mutate(Rated_date = round_date(as_datetime(timestamp), unit = "week"))
validation <- validation %>% mutate(Rated_date = round_date(as_datetime(timestamp), unit = "week"))

###############################################################################################################
#                          Step4:  Define RMSE function----
###############################################################################################################
#                          Step4:  Breif Summary Of this Step
###############################################################################################################
# 1) Define RMSE function
# 2) Create a table to document results
###############################################################################################################
#                          Step4:  R Code
###############################################################################################################


# Define RMSE function

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Create Table to track RMSE Results
rmse_target <- 0.86490
rmse_tracking <- data.frame(Method = "Project Goal RMSE", RMSE = "0.86490", Diff_Wrt_Goal = "-")

###############################################################################################################
#                          Step5:  Movie Recommendation Model - Optimization using Cross Validation-----
###############################################################################################################
#                          Step5:  Breif Summary Of this Step
###############################################################################################################
#  A Movie Recommendation Mode that has been built to predict the movie ratings.
#  Look at the Movie Recommendation Model report and RMD files to look at how the algorithm 
#  was built step by step. Here the final regularized algorithm has been given.
#  Model optimization will be performed using cross validation.
# The following cross validation process is used to find the lambda which produces the best RMSE value, 
# this optimized lambda value would gives us lowest RMSE. 
# 
# 1) 10 splits of the edx dataset were taken, where each split is a random sampling of data 
#    with 80% train data and 20% test data.
# 2) For each split or fold of data the following steps were repeated,
#   i)  A range of potential lambda values for turning the model were taken. 
#       Lambda range 4 to 6 with 0.1 increments. Here are the values of lambda used
#       4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2
#   ii) For each lambda value in the range, the following steps were performed
#     a) Calculate the Movie, User , Genre , Year and Release Date effects 
#     b) Predict the test_set ratings using the effects calculated in the above step
#     c) calculate RMSE for these predictions
#     d) store the lambda value and it's respective RMSE
#         iii) By the end of the above step we get RMSE values for for all lambda values in the set
#         iv)  Pick the lambda with minimum RMSE from the above set.
#         iv) The lambda that we obtained is the optimal lambda value for one particular split. 
# 3) By the end of previous step(step 2) we would get a set of lambda values and their               
#    respective RMSE values for all the splits.
# 4) We pick the lambda that has occurred most,that is the lambda with most frequency,
#    and this lambda value would be considered as an optimal parameter for the regularized         
#    model.
###############################################################################################################
#                         Step5:  R Code . This step is made into 3 parts due to its complexity
#                         Here is the outline of the code
#                         Step5.1:  R Code for making splits for Cross Validation
#                         Step5.2:  R Code for running the Cross Validation
#                         Step5.3:  R Code for picking the optimal lambda from the Cross Validation results
###############################################################################################################

###############################################################################################################
#                         Step5.1:  R Code for making splits for Cross Validation----
#                         10 splits , with lambda values - 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2
###############################################################################################################
# Set seed so that results would be same

set.seed(275)

# Now make 10 splits/folds of data. The below function returns a list with 10 folds of data
# each item in the list is a set of indexs generated for a single fold.

test_index_list <- createDataPartition(y = edx$rating, times = 10, p = 0.2, list = TRUE)
folds <- 1:10

## Set lambda values
inc <- 0.1
lambdas <- seq(4.5,5.2, inc)

###############################################################################################################
#                         Step5.2:  R Code for running the Cross Validation----
#                         This code will run cross validation and saves the cross validation results
#                         in the cross_validation_results
###############################################################################################################

# This function runs 10 times, in each time it takes it does the following
#         1) Get one split/fold data index's as generated by the above step
#         2) Build Train_set and Test_set datasets
#         3) Get all the lambda values and runs the below steps for each lambda value
#            a) Uses the data in Train_set to calculate all the effects
#            b) Uses the effects calculated in step 3 to predict the ratings in test_set
#            c) calculates RMSE
#         4) Pics the lambda with lowest RMSE and returns as output
# 
#  The output of this step is to return optimal lambda value and its RMSE for all the splits

cross_validation_results <- sapply(folds,function(fold){
  
###########   Build Train and Test Sets for the current fold #######################

  # Get all the test_index values for the current fold
  test_index <- test_index_list[fold]
  # Build train_set
  test_index <- test_index[[1]]
  train_set <- edx[-test_index,]
  temp <- edx[test_index,]
  # Build test_set
  # Make sure userId and movieId in test set are also in train set
  test_set <- temp %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  # Add rows removed from validation set back into train set
  removed <- anti_join(temp, test_set)
  train_set <- rbind(train_set, removed)
  rm(test_index, temp, removed)

###########   Run Algorithm on Each Fold                    ########################

# The below code would calculate rmse of the model for every value of lambda
# in the current fold
  
  rmses <- sapply(lambdas,function(l){

###########   Calculate Movie, User , Genre , Year and Movie Rated Date Effects ####
#             On the current fold using the lambda value passed
#  Below is the Movie Recommendation Model Built and the detailed explanation on how this was built is
#  given in the Movie_Recommendation_Model.pdf and rmd files.

    ###########   Calculate Mean Rating
    mu <- mean(train_set$rating)
    ###########   Calculate Movie Effect
    b_i <- train_set %>%
      group_by(movieId) %>%
      summarise(b_i = sum(rating - mu)/(n()+l))
    ###########   Calculate User Effect
    b_u <- train_set %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarise(b_u = sum(rating - b_i - mu)/(n()+l))
    ###########   Calculate Genre Effect
    b_g <- train_set %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      group_by(genres) %>%
      summarise(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
    ###########   Calculate Movie Release Year Effect
    b_y <- train_set %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      left_join(b_g, by="genres") %>%
      group_by(year) %>%
      summarise(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
    ###########   Calculate Movie Rated Dated Effect
    b_r <- train_set %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      left_join(b_g, by="genres") %>%
      left_join(b_y, by="year") %>%
      group_by(Rated_date) %>%
      summarise(b_r = sum(rating - b_i - b_u - b_g - mu)/(n()+l))

###########   Predict ratings based on the above calculated effects ####
#             On the current fold 

    ###########   Use all the effects during traing and predict the ratings
    predicted_ratings <- test_set %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      left_join(b_g, by="genres") %>%
      left_join(b_y, by="year") %>%
      left_join(b_r, by="Rated_date") %>%
      mutate(pred = mu + b_i + b_u + b_g + b_y + b_r) %>%
      pull(pred)

###########   Calculate RMSE On the current fold ###################################

    return(RMSE(predicted_ratings, test_set$rating))
  })
  
###########   Pick Optimal lambda as the one with lowest RMSE in the current split/fold
  
  # Assign optimal tuning parameter (lambda)
  lambda <- lambdas[which.min(rmses)]
  # Minimum RMSE achieved
  regularised_rmse <- min(rmses) 
  return_val <- c(fold,lambda,regularised_rmse)
  return(return_val)
})

####################################################################################################
#             Step5.3:  R Code for picking the optimal lambda from the Cross Validation Results----
####################################################################################################


# Wrangle data to get the cross validation results into a data frame

cv_results_df <- t(cross_validation_results)
colnames(cv_results_df) <- c("Split","lambda","RMSE")
row.names(cv_results_df) <- NULL

# Get the frequencies of lambds from the cross validation table
lambda_frequencies <- data.frame(cv_results_df) %>% group_by(lambda) %>% summarize( frequency = n())

# Get the lambda with max frequency and this is the optimal lambda value for the model
max_lambda_index <-which.max(lambda_frequencies$frequency)
optimal_lambda <- lambda_frequencies$lambda[max_lambda_index]
# Get the RMSE of the optimized model with the trained data
regularized_model_rmse <- data.frame(cv_results_df) %>% filter( lambda == optimal_lambda) %>%
  summarize( avg = mean(RMSE)) %>% pull(avg)

# Document the results of the trained algorithm

rmse_tracking <- rmse_tracking %>% rbind(c("Optimized - Movie Recommendation Model - Performance in Training", 
                                           round(regularized_model_rmse,5), 
                                           round(regularized_model_rmse-rmse_target,5)))

###############################################################################################################
#                          Step6:  Movie Recommendation Model - Validation & Results-----
###############################################################################################################
#                          Step6:  Breif Summary Of this Step
###############################################################################################################
# Now that the machine learning model is built and optimized , 
# and it's RMSE is  below the target target RMSE of 0.86490, 
# that means we have achieved the target RMSE in the training dataset, 
# we are ready to run and see how our final regularized Movie Recommendation Model 
# predicts the ratings in the validation dataset.
# 
# Here is the process followed to perform the validation
# 
#   1) Use the optimal lambda value that we got in the previous section, that is `r optimal_lambda`
#   2) Use the regularized Movie Recommendation Model on the entire edx dataset to calculate the 
#     Movie, User , Genre , Year and Movie Rated Date effects. 
#    Note: While building and optimizing the model we only used train_set, 
#    which is only 80% of the edx dataset, for calculating the effects, 
#    now as the model is finalized, we take the entire dataset, 
#    so that the model gets to use more data to calculate the effects.
#   3) Use the Movie, User, Genre , Year and Rated Date effects calculated in step 2
#      to predict the user ratings in the validation dataset.
#   4) Calculate the RMSE
#   5) Save the Model RMSE to the RMSE tracking table and print the output.
###############################################################################################################
#                         Step6:  R Code
###############################################################################################################

# Set the optimal lambda value that was obtained in the above optimization step

l = optimal_lambda

###########   Calculate Movie, User , Genre , Year and Movie Rated Date Effects ####
#             On the using the entire edx dataset and the optimal lambda value
####################################################################################

###########   Calculate Mean Rating
mu <- mean(edx$rating)

###########   Calculate Movie Effect
b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n()+l))

###########   Calculate User Effect
b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/(n()+l))

###########   Calculate Genre Effect
b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarise(b_g = sum(rating - b_i - b_u - mu)/(n()+l))

###########   Calculate Movie Release Year Effect
b_y <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(year) %>%
  summarise(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))

###########   Calculate Movie Rated Date Effect
b_r <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="year") %>%
  group_by(Rated_date) %>%
  summarise(b_r = sum(rating - b_i - b_u - b_g - mu)/(n()+l))

###########   Predict ratings based on the above calculated effects ####

predicted_ratings <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="year") %>%
  left_join(b_r, by="Rated_date") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_r) %>%
  pull(pred)

###########   Calculate RMSE ###################################----

valiation_rmse <- RMSE(predicted_ratings, validation$rating)

#  Document the results in a table and output the results to the console

# Add the movie release year effect to RMSE tracking table and print the output
rmse_tracking <- rmse_tracking %>% rbind(c("Optimized Movie Recommendation Model - Performance with Validation Dataset", 
                                           round(valiation_rmse,5), 
                                           round(valiation_rmse-rmse_target,5)))

rmse_tracking %>% as_tibble()


print("Final Model Validation Results")

print_lambda <- paste("Optimal Lambda Used for Training the final algorithm -",optimal_lambda)
print(print_lambda)
print_validation_rmse <- paste("Movie Recommendation Model - Ouput Result After validating the data in validation dataset -",valiation_rmse)
print(print_validation_rmse)
diff <- valiation_rmse - rmse_target
print(paste(" Hurrey - Movie Recommendation Model has beaten the target RMSE by -",
            diff, " and this means validation RMSE is ",abs(diff)," less than Target RMSE  "))




