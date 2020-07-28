# Use Random Forest to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

fraud_data <- read.csv(file.choose())
str(fraud_data)
summary(fraud_data)

library(moments)
hist(fraud_data$Taxable.Income)
boxplot(fraud_data$Taxable.Income)

## Now will categorize the data as taxable_income <= 30000 as "Risky" and others are "Good"

risk_good <- ifelse(fraud_data$Taxable.Income<=30000,"Risky","Good")
data_fraud <- data.frame(fraud_data,risk_good)

## Now will aplit data into train and test

fraud_train_data <- data_fraud[1:300,]
fraud_test_data <- data_fraud[301:600,]

## now will build first model by using random forest

install.packages("randomForest")
library(randomForest)
attach(fraud_train_data)

## now we will build our model on train

fraud_train_model <- randomForest(risk_good ~ . - risk_good,data = fraud_train_data,ntree = 30,importance= TRUE)

pred_fraud_train <- fraud_train_model$predicted

table(pred_fraud_train,fraud_train_data$risk_good)

# pred_fraud_train Good Risky
#        Good      229     1
#        Risky       1    69

mean(fraud_train_data$risk_good==pred_fraud_train) ## accuracy 0.9933333

## now will predict data using test

pred_fraud_test <- predict(fraud_train_model,newdata = fraud_test_data[,-7]) ## exclude risk_good column

table(pred_fraud_test,fraud_test_data$risk_good)

#          Reference
# Prediction  Good Risky
#    Good     246     0
#    Risky     0     54

mean(pred_fraud_test==fraud_test_data$risk_good) ## 100 % accuracy

## lets plot confusion matrix for that call caret library

library(lattice)
library(ggplot2)
library(caret)

confusionMatrix(fraud_train_data$risk_good,fraud_train_model$predicted)
confusionMatrix(fraud_test_data$risk_good,pred_fraud_test)

## lets visualize the plot

plot(fraud_train_model,lwd=3)
legend("topright",colnames(fraud_train_model$err.rate),col = 1:4,cex = 0.35,fill = 1:10)

## Graph Observation :

# taxable income which is less than 30k i.e. risky has very fluctuating line till 18
# after that 18 to 30 we got error free values.
# taxable income which we have classify as good shows errors till 5,then we got
# straight line i.e. error free results

## Now lets build model using tuneRF

fraud_tune_train <- tuneRF(fraud_train_data[,-7],fraud_train_data[,7],ntreeTry = 40,
                           stepFactor = 0.5,plot = TRUE,trace = TRUE,improve = 0.05)

# mtry = 2  OOB error = 0.33% 
# Searching left ...
# mtry = 4 	OOB error = 0.33% 
# 0 0.05 
# Searching right ...
# mtry = 1 	OOB error = 5.33% 
# -15 0.05 

## now will use this mtry values to build our model

attach(fraud_train_data)

rf_train_fraud <- randomForest(risk_good ~ . -risk_good,data = fraud_train_data,
                               mtry = 2,importance = TRUE,proximity = TRUE)
rf_train_fraud

# OOB estimate of  error rate: 0.33%
# Confusion matrix:
#       Good Risky   class.error
# Good   229     1   0.004347826
# Risky    0    70   0.000000000

rf_pred <- predict(rf_train_fraud,fraud_train_data)
confusionMatrix(rf_pred,fraud_train_data$risk_good)

#         Reference
#   Prediction Good Risky
#       Good   230     0
#       Risky    0    70

#Accuracy : 1

rf_test_fraud <- predict(rf_train_fraud,fraud_test_data)
confusionMatrix(rf_test_fraud,fraud_test_data$risk_good)

#       Reference
# Prediction Good Risky
#     Good   246     0
#     Risky    0    54

# Accuracy : 1