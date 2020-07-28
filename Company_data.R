#Random Forest

#Assignment


#About the data: 
#Let’s consider a Company dataset with around 10 variables and 400 records. 
#The attributes are as follows: 
# Sales -- Unit sales (in thousands) at each location
# Competitor Price -- Price charged by competitor at each location
# Income -- Community income level (in thousands of dollars)
# Advertising -- Local advertising budget for company at each location (in thousands of dollars)
# Population -- Population size in region (in thousands)
# Price -- Price company charges for car seats at each site
# Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
# Age -- Average age of the local population
# Education -- Education level at each location
# Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
# US -- A factor with levels No and Yes to indicate whether the store is in the US or not
#The company dataset looks like this: 
#Problem Statement:
#A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
#Approach - A Random Forest can be built with target variable Sales (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

company_data <- read.csv(file.choose())
summary(company_data)
str(company_data)

library(moments)
hist(company_data$Sales)
boxplot(company_data$Sales)

## now will categorize the sales 

sales_cat <- ifelse(company_data$Sales < 8,"No","Yes")
data_company <- data.frame(company_data,sales_cat)

table(data_company$sales_cat)
#  No Yes 
# 236 164

## now will split data into train and test

company_train_data <- data_company[1:200,]
company_test_data <- data_company[201:400,]

## lest build our first model using random forest on training dataset 

library(randomForest)
attach(company_train_data)

company_train_model <- randomForest(sales_cat ~ . - sales_cat,data = company_train_data,ntree = 35,importance = TRUE)

## take y = sales_cat in which our sales categories are available and pass train data

pred_train <- company_train_model$predicted

head(pred_train)
# 1   2   3   4   5   6 
# Yes Yes Yes  No  No Yes 
#Levels: No Yes

## predicted train values

table(pred_train,company_train_data$sales_cat)

#pred_train   No  Yes
#        No   117   1
#        Yes   0  82

## Lets try to evaluate the performance of model

mean(company_train_data$sales_cat == pred_train)

## 0.995 is the accuracy for model

# now we will predict data using test data set

pred_test <- predict(company_train_model,newdata = company_test_data)

table(pred_test,company_test_data$sales_cat)

#pred_test   No Yes
#      No   119   1
#      Yes   0  81

mean(company_test_data$sales_cat == pred_test)
## accuracy for this model is 100

library(ggplot2)
library(lattice)
library(caret)

confusionMatrix(company_train_data$sales_cat,company_train_model$predicted)
confusionMatrix(company_test_data$sales_cat,pred_test)

## lets visualize the plot

plot(company_train_model,lwd=3)
legend("topright",colnames(company_train_model$err.rate),col = 1:4,cex = 0.35,fill = 1:10)

### from this graph we can say that,there are lot of errors in Yes .From 24 to 35
## we got error free result and this our sale greater than 8.For sale less than 8
## has 17 to 35 results as error free.

## now lets build Tune Random Forest Model mtry
pre
? tuneRF
## tuneRF(x, y, mtryStart, ntreeTry=50, stepFactor=2, improve=0.05,
## trace=TRUE, plot=TRUE, doBest=FALSE, ...)

tune_train_forest <- tuneRF(company_train_data[,-12],company_train_data[,12],stepFactor = 0.5,
                            plot = TRUE,ntreeTry = 24,trace = TRUE,improve = 0.05)

## mtry = 3  OOB error = 1% 
## Searching left ...
## mtry = 6 	OOB error = 1% 
## 0 0.05 
## Searching right ...
## mtry = 1 	OOB error = 14.5% 
## -13.5 0.05 

## now lets build RF model by using this mtry value

attach(company_train_data)

## here we rae just passing mtry value which we got in above after using tuneRF
## to get more accuracy on model

company_data_RF1 <- randomForest(sales_cat ~ . -sales_cat,data = company_train_data,ntree = 24,
                                 mtry =3,importance =TRUE,proximity = TRUE)

company_data_RF1

#OOB estimate of  error rate: 0.5%

# Confusion matrix:
#      No  Yes   class.error
#  No  116   1  0.008547009
#  Yes   0  83  0.000000000

## lets pred values for this train model

RF1_pred <- predict(company_data_RF1,company_train_data)
confusionMatrix(RF1_pred,company_train_data$sales_cat)

#        Reference
#Prediction  No Yes
#       No  117   0
#       Yes   0  83

#Accuracy : 1     

## Now will predict this values for test

RF_pred_test <- predict(company_data_RF1,company_test_data)
confusionMatrix(RF_pred_test,company_test_data$sales_cat)

#        Reference
# Prediction   No Yes
#        No   119   0
#        Yes    0  81

#Accuracy : 1 