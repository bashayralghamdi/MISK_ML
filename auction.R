#title: "auction"
#author: "Bashayr Alghamdi"
#date: "03/11/2020"

#analysis about auction of xbox game consoles on ebay
#https://www.modelingonlineauctions.com/datasets

#problem type: supervised regression
#response variable: Price (i.e.,28.0 - 501.8)
#features: 32
#observations: 119,390
#objective: use property attributes to predict the sale price of xbox game .

# load packages
library(tidyverse)
library(rsample)
library(caret)
library(recipes)
library(vip)
library(dplyr)
library(Metrics) 
library(pls)
library(glmnet)  



# Read in the data (csv format):
auctions7 <- readr::read_csv("data/Xbox 7-day auctions.csv")
glimpse(auctions7)
summary(auctions7)
auctions7$days <- 7

auctions5 <- readr::read_csv("data/Xbox 5-day auctions.csv")
glimpse(auctions5)
summary(auctions5)
auctions5$days <- 5

auctions3 <- readr::read_csv("data/Xbox 3-day auctions.csv")
glimpse(auctions3)
summary(auctions3)
auctions3$days <- 3

#merge the dataset
auctionmrg <- merge(auctions7,auctions5,by= c( "auctionid","bid","bidtime","bidder","bidderrate","openbid","price","days" ), all=TRUE)
glimpse(auctionmrg)

#finally this is our dataest
auctions <- merge(auctionmrg,auctions3,by= c( "auctionid","bid","bidtime","bidder","bidderrate","openbid","price","days" ), all=TRUE)


#get familiar with the data
glimpse(auctions)
summary(auctions)#there is missing value in children column

#is there any missing value on the target 
#Plot the missing values.
visdat::vis_miss(auctions_train, cluster = TRUE)
#there a few missing value in "bidder" and "biderrat" columns
sum(is.na(auctions))#the number of missing value is 27
#remove missing value
auctions <- auctions %>% 
  na.omit()

# initial dimension
dim(auctions)

# response variable
head(auctions)

#change the type of bidder from char to factor
auctions$bidder <- as.factor(auctions$bidder)
#save the new data as csv file
write.csv(auctions,"data/auctions.csv", row.names = FALSE)


#Split into training vs testing data
set.seed(123)
split  <- initial_split(auctions, prop = 0.7, strata = "price")
auctions_train  <- training(split)
auctions_test   <- testing(split)

# Do the distributions line up?  
ggplot(auctions_train, aes(x = price)) + 
  geom_line(aes(col="train"),
            stat = "density", 
            trim = TRUE,col = "black") + 
  geom_line(aes(col="test"),
            data = auctions_test, 
            stat = "density", 
            trim = TRUE, col = "red")



#feature engineering 
#plot the target
#Is the response skewed?
auctions_train%>% 
  ggplot(aes(price))+
  geom_histogram() #positive skwed 

#apply a transformation normalize
auctions%>% 
  ggplot(aes(log10(price)))+
  geom_histogram()

#relationship between price and bid
ggplot(aes(price,bid),data =auctions_train )+
  geom_point()+
  geom_smooth()




#Do any features have near-zero or zero variance
remove_cols <- nearZeroVar(auctions_train, saveMetrics= TRUE) 
remove_cols
#there are no features have near-zero or zero variance

#the numeric features
auctions_train%>% 
  ggplot(aes(auctionid))+
  geom_histogram()

auctions_train%>% 
  ggplot(aes(bid))+
  geom_histogram()

auctions_train%>% 
  ggplot(aes(bidtime))+
  geom_histogram()

auctions_train%>% 
  ggplot(aes(bidderrate))+
  geom_histogram()

auctions_train%>% 
  ggplot(aes(openbid))+
  geom_histogram()


#the categorical features.
#Error in { : 
#task 1 failed - "Column 4 of x is of class character while 
#matching column 4 of y is of class factor"

auctions_train%>% 
  ggplot(aes(bidder))+
  geom_bar()

lump <- auctions_train %>% 
  select(bidder) %>% 
  group_by(bidder) %>% 
  mutate(count=n()) %>% 
  arrange(count) 

min(lump$count) #minimum number of counting levels 
max(lump$count) #maximum number of counting levels 



blueprint <- recipe(price ~ ., data = auctions_train) %>%
  step_dummy(bidder,one_hot = TRUE) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) 


# create a resampling method
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 5
)

# create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(1, 50, by = 2))


knn_fit_bp <- train(
  blueprint, 
  data = auctions_train, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "RMSE"
)

# execute grid search with knn model
#    use RMSE as preferred metric
knn_fit <- train(
  price ~ .,
  data = auctions_train, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "RMSE"
)


# evaluate results
# print model results
knn_fit_bp
knn_fit

# plot cross validation results
ggplot(knn_fit$results, aes(k, RMSE)) + 
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = scales::dollar)

ggplot(knn_fit_bp$results, aes(k, RMSE)) + 
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = scales::dollar)

vip(knn_fit)
vip(knn_fit_bp)


# linear regression model
#include all possible main effects

set.seed(123)
cv_model <- train(
  price ~ .,
  data = auctions_train, 
  method = "lm",
  trControl = cv
)

set.seed(123)
cv_model_bp <- train(
  blueprint, 
  data = auctions_train, 
  method = "lm",
  trControl = cv
)

p <- length(auctions_train) - 1
hyper_grid_p <- expand.grid(ncomp = seq(2, 100, length.out = 10))

#pcr
set.seed(123)
cv_pcr <- train(
  price ~ .,
  data = auctions_train, 
  trControl = cv,
  method = "pcr",
  preProcess = c("center", "scale"),
  tuneGrid = hyper_grid_p,
  metric = "RMSE"
)
cv_pcr$results %>%
  filter(ncomp == as.numeric(cv_pcr$bestTune))


#pls

set.seed(123)
cv_pls <- train(
  price ~ ., 
  data = auctions_train, 
  trControl = cv,
  method = "pls",
  preProcess = c("center", "scale"),
  tuneGrid = hyper_grid_p,
  metric = "RMSE"
)

cv_pls$results %>%
  filter(ncomp == as.numeric(cv_pls$bestTune))

vip(cv_pls)

#regularized 
X <- model.matrix(price ~ ., auctions_train)[, -1]

Y <- log(auctions_train$price)

hyper_grid_g <- expand.grid(
  alpha = seq(0, 1, by = .25),
  lambda = c(0.1, 10, 100, 1000, 10000)
)

ridge <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)

plot(ridge, xvar = "lambda")

lasso <- glmnet(
  x = X,
  y = Y,
  alpha = 1
)

plot(lasso, xvar = "lambda")






# perform resampling
set.seed(123)
cv_glmnet <- train(
  x = X,
  y = Y,
  method = "glmnet",
  preProc = c( "center", "scale"),
  trControl = cv,
  tuneGrid = hyper_grid_g,
  tuneLength = 10
)


cv_glmnet$results %>%
  filter(
    alpha == cv_glmnet$bestTune$alpha,
    lambda == cv_glmnet$bestTune$lambda
  )

vip(cv_glmnet, num_features = 10, geom = "point")

p1 <- pdp::partial(cv_glmnet, pred.var = "bid", grid.resolution = 20) %>%
  as_tibble() %>% 
  mutate(yhat = exp(yhat)) %>%
  ggplot(aes(bid, yhat)) +
  geom_line() +
  scale_y_continuous(limits = c(0, 400), labels = scales::dollar)

p2 <- pdp::partial(cv_glmnet, pred.var = "biddermregestr", grid.resolution = 20) %>%
  as_tibble() %>% 
  mutate(yhat = exp(yhat)) %>%
  ggplot(aes(biddermregestr, yhat)) +
  geom_line() +
  scale_y_continuous(limits = c(0, 300), labels = scales::dollar)   

p3 <- pdp::partial(cv_glmnet, pred.var = "bidderadavisa1", grid.resolution = 20) %>%
  as_tibble() %>% 
  mutate(yhat = exp(yhat)) %>%
  ggplot(aes(bidderadavisa1, yhat)) +
  geom_line() +
  scale_y_continuous(limits = c(0, 300), labels = scales::dollar)

p4 <- pdp::partial(cv_glmnet, pred.var = "biddersavant51", grid.resolution = 20) %>%
  as_tibble() %>% 
  mutate(yhat = exp(yhat)) %>%
  ggplot(aes(biddersavant51, yhat)) +
  geom_line() +
  scale_y_continuous(limits = c(0, 300), labels = scales::dollar)


grid.arrange(p1, p2, p3, p4, nrow = 2)





