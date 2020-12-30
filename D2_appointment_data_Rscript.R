set.seed(1)
setwd("/Users/neesharathee/Desktop/Neesha/data1")
##Step1 - Collecting the data
#importing data
appointment_data = read.csv("appointment_data.csv",stringsAsFactors = FALSE)

##Step2 - Exploring and preparing the data
#checking structure of data
summary(appointment_data)
str(appointment_data)

#checking for NAs in data
sum(is.na(appointment_data))
library(VIM)
mp <- aggr(appointment_data, col=c('navyblue','yellow'),
           numbers=TRUE, sortVars=TRUE,
           labels=names(appointment_data), cex.axis=.7,
           gap=3, ylab=c("Missing data","Pattern"))


#removing negative and zero aged data- noise
summary(appointment_data$Age)
library(dplyr)
appointment_data <- appointment_data %>%
  filter(appointment_data$Age > 0)

#transforming data
str(appointment_data)
appointment_data <- appointment_data[,-c(1,2)]
appointment_data$Gender <- factor(appointment_data$Gender)
appointment_data$Scholarship <- factor(appointment_data$Scholarship)
appointment_data$Neighbourhood <- factor(appointment_data$Neighbourhood)
appointment_data$Hipertension <- factor(appointment_data$Hipertension)
appointment_data$Diabetes <- factor(appointment_data$Diabetes)
appointment_data$Alcoholism <- factor(appointment_data$Alcoholism)
appointment_data$Handcap <- factor(appointment_data$Handcap)
appointment_data$SMS_received <- factor(appointment_data$SMS_received)
appointment_data$No.show <- factor(appointment_data$No.show)
appointment_data$ScheduledDay <- as.Date(appointment_data$ScheduledDay)
appointment_data$AppointmentDay <- as.Date(appointment_data$AppointmentDay)
appointment_data$dayofweek <- weekdays(appointment_data$AppointmentDay)
table(appointment_data$dayofweek)
appointment_data$dayofweek <-factor(appointment_data$dayofweek)
appointment_data$awaiting_time <- appointment_data$AppointmentDay - appointment_data$ScheduledDay
appointment_data$awaiting_time <- as.numeric(appointment_data$awaiting_time)
appointment_data <- appointment_data %>%
  filter(appointment_data$awaiting_time > 0)
appointment_data <- appointment_data[,-c(2,3)]
str(appointment_data)
names(appointment_data)<-c("Sex","Age","Area","Govt_funded","Hypertension","Diabetes","Alcoholism","Handicap","Follow_up_SMS","No_show","Awaiting_time","Day")

#visualising No_show wrt predictors
library(ggplot2)
library(ggthemes)
ggplot(appointment_data, aes(x = Sex, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Sex') + theme_few()
ggplot(appointment_data, aes(x = Age, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Age') + theme_few()
ggplot(appointment_data, aes(x = Area, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Area') + theme_few()
ggplot(appointment_data, aes(x = Govt_funded, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Govt_funded') + theme_few()
ggplot(appointment_data, aes(x = Hypertension, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Hypertension') + theme_few()
ggplot(appointment_data, aes(x = Diabetes, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Diabetes') + theme_few()
ggplot(appointment_data, aes(x = Alcoholism, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Alcoholism') + theme_few()
ggplot(appointment_data, aes(x = Handicap, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Handicap') + theme_few()
ggplot(appointment_data, aes(x = Follow_up_SMS, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Follow_up_SMS') + theme_few()
ggplot(appointment_data, aes(x = Day, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Day') + theme_few()
ggplot(appointment_data, aes(x = Awaiting_time, fill = factor(No_show))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Awaiting_time') + theme_few()

#stratified random sampling
library(caret)
index <-createDataPartition(appointment_data$No_show,p=0.75,list = FALSE)
appointment_train <- appointment_data[index, ]
appointment_test <- appointment_data[-index, ]
row.names(appointment_train) <- 1:52372
row.names(appointment_test) <- 1:17456
table(appointment_train$No_show)
table(appointment_test$No_show)



#Step3 training a model
###########1- NaiveBayes
library(e1071)

str(appointment_train)
appointment_train_labels <- appointment_train[,10]
appointment_train <- appointment_train[,-10]
appointment_test_labels <- appointment_test[,10]
appointment_test <- appointment_test[,-10]

#model1 
appointment_model_n <- naiveBayes(appointment_train,appointment_train_labels)
summary(appointment_model_n)
app_predict_n <- predict(appointment_model_n,appointment_test)
confusionMatrix(app_predict_n,appointment_test_labels)

#model2 
appointment_model_nl <- naiveBayes(appointment_train,appointment_train_labels,laplace = 1)
summary(appointment_model_nl)
app_predict_nl <- predict(appointment_model_nl,appointment_test)
confusionMatrix(app_predict_nl,appointment_test_labels)

#ROC curve
app_pred <- predict(appointment_model_nl,appointment_test, type = "raw")
pred = prediction(app_pred[,2],appointment_test_labels)
pref = performance(pred,"tpr","fpr")
plot(pref, avg= "threshold", colorize=T, lwd=3, main="ROC curve for naive bayes")
abline(a=0,b=1,lwd=2,lty=2,col="grey")

#AUC
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc

#F score
result <- confusionMatrix(app_predict_nl,appointment_test_labels,mode="prec_recall")
result$byClass["Precision"]
result$byClass["Recall"]
result$byClass["F1"]

#######2- random forest- 0.7152, kappa-0.012
#model1
library(randomForest)
#removing area as random forest can't handle predictors with more than 53 categpries
appointment_train_rf <- appointment_train[,-3]
rf <- randomForest(appointment_train_rf,appointment_train_labels,ntree=500,mtry=sqrt(10))
app_predict <- predict(rf,appointment_test)
confusionMatrix(app_predict,appointment_test_labels)

#model2-Accuracy    
rf2 <- randomForest(appointment_train_rf,appointment_train_labels,mtry=8)
app_predict <- predict(rf2,appointment_test)
confusionMatrix(app_predict,appointment_test_labels)
#model3-Accuracy  
rf3 <- randomForest(appointment_train_rf,appointment_train_labels,mtry=4)
app_predict <- predict(rf3,appointment_test)
summary(app_predict)
confusionMatrix(app_predict,appointment_test_labels)
#ROC curve
app_pred <- predict(rf3,appointment_test, type = "prob")
pred = prediction(app_predict[,2],appointment_test_labels)
pref = performance(pred,"tpr","fpr")
plot(pref, avg= "threshold", colorize=T, lwd=3, main="ROC curve for random forest")
abline(a=0,b=1,lwd=2,lty=2,col="grey")

#AUC
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc

#F measure
result <- confusionMatrix(app_predict,appointment_test_labels,,mode="prec_recall")
result$byClass["Precision"]
result$byClass["Recall"]
result$byClass["F1"]

#ROC curve
app_pred <- predict(rf3,appointment_test, type = "prob")
pred = prediction(app_pred[,2],appointment_test_labels)
pref = performance(pred,"tpr","fpr")
app_pred_n <- predict(appointment_model_nl,appointment_test, type = "raw")
pred_n = prediction(app_pred_n[,2],appointment_test_labels)
pref_n = performance(pred_n,"tpr","fpr")
plot(pref_n, avg= "threshold", colorize=T, lwd=3, main="ROC curve RF Vs NB")
plot(perf, col=1, add=TRUE)
plot(pref_n, col=2, add=TRUE)
legend(0.6, 0.6, c("rforest","NaiveBayes"), 1:2)
abline(a=0,b=1,lwd=2,lty=2,col="grey")

########### decision tree --- rejected
library(tree)
str(appointment_train_d)
appointment_train_d <- appointment_train
appointment_train_d$No_show <- appointment_train_labels
appointment_test_d <- appointment_test
appointment_test_d$No_show <- appointment_test_labels
#removing area as tree can't handle predictors with more than 32 categpries
appointment_train_d <- appointment_train[,-3]
appointment_test_d <- appointment_test[,-3]
tree.app_data <- tree(No_show~.,appointment_train_d)
summary(tree.app_data)
plot(tree.app_data)
text(tree.app_data,pretty = 0)

library(C50)
app_model_c50 <- C5.0(appointment_train[,-3],appointment_train_labels)
summary(app_model_c50)
plot(app_model_c50)
app_predict <- predict(app_model_c50,appointment_test)
confusionMatrix(app_predict,appointment_test_labels)

##rpart
library(rpart)
app.rpart <- rpart(No_show~.,data = appointment_train_d,method = "class",control = rpart.control(minsplit = 1,minbucket = 1,cp=0))
plot(app.rpart,margin = 0.1)
text(app.rpart,use.n = T,pretty = T,cex=0.8)
pred = predict(app.rpart,appointment_test_d,type="class")
confusionMatrix(pred,appointment_test$No_show)

#################################################

