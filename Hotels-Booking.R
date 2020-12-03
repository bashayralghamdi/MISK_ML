#title: "Hotels"
#author: "Bashayr Alghamdi"
#date: "03/11/2020"
#analysis about hotels booking

#Hotels booking dataset comes from an open hotel booking from [tidy tuesday]
#demand dataset from Antonio,Almeida and Nunes, 2019
#problem type: supervised binomial classification
#response variable: is_cancel (i.e., 0 and 1)
#features: 32
#observations: 119,390
#objective: use customer's reservation state if they will cancel or not.

# load packages
library(tidyverse)
library(rsample)
library(caret)
library(recipes)
library("ROCR")
library(vip)
library(dplyr)

# Read in the data (csv format):
hotels <- readr::read_csv("data/hotels.csv")

#get familiar with the data
spec(hotels)
summary(hotels)#there is missing value in children column

#missing value
sum(is.na(hotels))#the number of missing value is 4

#remove missing value
hotels <- hotels %>% 
  na.omit()

# initial dimension
dim(hotels)

# response variable
head(hotels$is_canceled)

#convert the target from num to factor
hotels$is_canceled <- as.factor(hotels$is_canceled)

#plot the target / response variable
hotels %>% 
ggplot(aes(is_canceled))+
  geom_bar()
hotels %>% 
  group_by(is_canceled) %>% 
  mutate(n=n())

  

hotels <- hotels %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)

# split data
set.seed(123)
hotels_split <- initial_split(hotels,prop = 0.7, strata = "is_canceled")
hotels_train <- training(hotels_split)
hotels_test <- testing(hotels_split)

dim(hotels_split)


#apply a blueprint of feature engineering processes
blueprint <- recipe(is_canceled ~ ., data = hotels_train) %>%
  step_nzv(all_nominal()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%

  
  
#apply the model to your data without pre-applying feature engineering processes.
  
model <- glm(is_canceled ~ .,family = "binomial", data = hotels)#take long time I will not use it
summary(model)
  
model1 <- glm(is_canceled ~ lead_time+adults+is_repeated_guest+previous_bookings_not_canceled+booking_changes,family = "binomial", data = hotels)
summary(model1)
model2 <- glm(is_canceled ~ lead_time+adults+previous_bookings_not_canceled,family = "binomial", data = hotels)
summary(model2)
model3 <- glm(is_canceled ~ lead_time+adults,family = "binomial", data = hotels)
summary(model3)
model4 <- glm(is_canceled ~ lead_time,family = "binomial", data = hotels)
summary(model4)

# Coefficients
tidy(model1)
tidy(model2)
tidy(model3)
tidy(model4)

exp(coef(model1))
exp(coef(model2))
exp(coef(model3))
exp(coef(model4))

#feature engineering:
hotels %>% 
ggplot(aes(lead_time))+ # positive skewed
  geom_bar()

hotels %>% 
  ggplot(aes(adults))+
  geom_bar()

hotels %>% 
  ggplot(aes(previous_bookings_not_canceled))+# positive skewed
  geom_bar()

hotels %>% 
  ggplot(aes(is_repeated_guest))+
  geom_bar()

hotels %>% 
  ggplot(aes(booking_changes))+ # positive skewed
  geom_bar()

# model1
blueprint <- recipe(is_canceled ~ lead_time+adults+is_repeated_guest+previous_bookings_not_canceled+booking_changes, data = hotels_train) %>%
  step_nzv(all_numeric()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

set.seed(123)
cv_model1 <- train(
  is_canceled ~ lead_time+adults+is_repeated_guest+previous_bookings_not_canceled+booking_changes,
  data = hotels_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model2 <- train(
  is_canceled ~ lead_time+adults+previous_bookings_not_canceled,
  data = hotels_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model3 <- train(
  is_canceled ~ lead_time+adults,
  data = hotels_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)


set.seed(123)
cv_model4 <- train(
  is_canceled ~ lead_time,
  data = hotels_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)


# accuracy
summary(resamples(list(model1=cv_model1,
                       model2=cv_model2,
                       model3=cv_model3,
                       model4=cv_model4)))$statistics$Accuracy 


pred_class <- predict(cv_model1, hotels_train)

# create confusion matrix
confusionMatrix(
  data = relevel(pred_class, ref = "1"), 
  reference = relevel(hotels_train$is_canceled, ref = "1")
)



m1_prob <- predict(cv_model1, hotels_train, type = "prob")$Yes
m4_prob <- predict(cv_model4, hotels_train, type = "prob")$Yes



### Is not working
#Error: Format of predictions is invalid. It couldn't be coerced to a list.
# Compute AUC metrics for cv_model1 and cv_model4
perf1 <- prediction(m1_prob, hotels_train$is_canceled) %>%
  performance(measure = "tpr", x.measure = "fpr")
perf4 <- prediction(m4_prob, hotels_train$is_canceled) %>%
  performance(measure = "tpr", x.measure = "fpr")

# Plot ROC curves for cv_model1 and cv_model3
plot(perf1, col = "black", lty = 2)
plot(perf4, add = TRUE, col = "blue")
legend(0.8, 0.2, legend = c("cv_model1", "cv_model3"),
       col = c("black", "blue"), lty = 2:1, cex = 0.6)


