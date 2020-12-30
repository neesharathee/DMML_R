##Step1 - Collecting the data
#importing test and train datasets

setwd("D:\\Neesha\\data3")
sign_mnist_train = read.csv("sign_mnist_train.csv")
sign_mnist_test = read.csv("sign_mnist_test.csv")

##Step2 - Exploring and preparing the data
#checking structure of data
str(sign_mnist_train)
summary(sign_mnist_train)
str(sign_mnist_test)

#checking for NAs in both datasets
library(VIM)
mp <- aggr(sign_mnist_train, col=c('navyblue','yellow'),
           numbers=TRUE, sortVars=TRUE,
           labels=names(sign_mnist_train), cex.axis=.7,
           gap=3, ylab=c("Missing data","Pattern"))

sum(is.na(sign_mnist_train))
sum(is.na(sign_mnist_test))

#converting label to factor in both datasets
sign_mnist_train$label <- as.factor(sign_mnist_train$label)
sign_mnist_test$label <- as.factor(sign_mnist_test$label)
table(sign_mnist_train$label)
table(sign_mnist_test$label)

##Step3 - Training a model on the data
#1- svm: one-to-many

#model no 1 [acc:0.8053542 ]
library(kernlab)
set.seed(1)
sign_classifier <- ksvm(label~.,data=sign_mnist_train,kernel="vanilladot")
sign_classifier

##Step4 - Evaluating model performance
sign_predictions <- predict(sign_classifier,sign_mnist_test)
head(sign_predictions)
table(sign_predictions,sign_mnist_test$label)
agreement <- sign_predictions==sign_mnist_test$label
table(agreement)
#accuracy
prop.table(table(agreement))

#model no 2 [acc:0.8436977  ]
##Step5 - Improving model performance
sign_classifier_rbf <- ksvm(label~.,data=sign_mnist_train,kernel="rbfdot")
sign_classifier_rbf
sign_predictions_rbf <- predict(sign_classifier_rbf,sign_mnist_test)
head(sign_predictions_rbf)
table(sign_predictions_rbf,sign_mnist_test$label)
agreement <- sign_predictions_rbf==sign_mnist_test$label
table(agreement)
#accuracy
prop.table(table(agreement))

#model no 3 [acc:0.8473229 ]
sign_classifier_rbf_c5 <- ksvm(label~.,data=sign_mnist_train,kernel="rbfdot",C=5)
sign_predictions_rbf_c5 <- predict(sign_classifier_rbf_c5,sign_mnist_test)
head(sign_predictions_rbf_c5)
table(sign_predictions_rbf_c5,sign_mnist_test$label)
agreement <- sign_predictions_rbf_c5==sign_mnist_test$label
table(agreement)
#accuracy
prop.table(table(agreement))

#model no 4 [acc:0.8480201 ]
sign_classifier_rbf_c10 <- ksvm(label~.,data=sign_mnist_train,kernel="rbfdot",C=10)
sign_predictions_rbf_c10 <- predict(sign_classifier_rbf_c10,sign_mnist_test)
head(sign_predictions_rbf_c10)
table(sign_predictions_rbf_c10,sign_mnist_test$label)
agreement <- sign_predictions_rbf_c10==sign_mnist_test$label
table(agreement)
#accuracy
prop.table(table(agreement))


#model no 5 [acc:0.8020078 ]
#10-fold cross validation for choosing best value of tuning parameter - linear
set.seed(1)
library(e1071)
tune.out = tune(svm,label~.,data=sign_mnist_train,kernel="linear",
                ranges=list(cost=c(0.5,1,1.5)))
    summary(tune.out)
bestmod_l=tune.out$best.model
summary(bestmod_l)
sign_predictions_bestmod_l <- predict(bestmod_l,sign_mnist_test)
table(sign_predictions_bestmod_l,sign_mnist_test$label)
agreement <- sign_predictions_bestmod_l==sign_mnist_test$label
table(agreement)
#accuracy
prop.table(table(agreement))

#model no 6 [acc:0.8396542 ]
sign_classifier_rbf_c0.5 <- ksvm(label~.,data=sign_mnist_train,kernel="rbfdot",C=0.5)
sign_predictions_rbf_c0.5 <- predict(sign_classifier_rbf_c0.5,sign_mnist_test)
head(sign_predictions_rbf_c0.5)
table(sign_predictions_rbf_c0.5,sign_mnist_test$label)
agreement <- sign_predictions_rbf_c0.5==sign_mnist_test$label
table(agreement)
#accuracy
prop.table(table(agreement))

#Final evaluation- model no-4

library(caret)
confusionMatrix(sign_predictions_rbf_c10,sign_mnist_test$label)

#kappa statistic
library(vcd)
Kappa(table(sign_predictions_rbf_c10,sign_mnist_test$label))



#########################2- kNN
sign_mnist_train_knn <- sign_mnist_train[,-1]
sign_mnist_train_knn_labels <- sign_mnist_train[,1]
sign_mnist_test_knn <- sign_mnist_test[,-1]
sign_mnist_test_knn_labels <- sign_mnist_test[,1]
library(class)
#model1 [acc-0.6071, Kappa : 0.589]
sign_classifier_knn <- knn(train = sign_mnist_train_knn ,test = sign_mnist_test_knn,
                           sign_mnist_train_knn_labels,k=166)

confusionMatrix(sign_classifier_knn,sign_mnist_test$label)

#model2 [acc-0.8059,Kappa : 0.7969  ]
sign_classifier_knn_3 <- knn(train = sign_mnist_train_knn ,test = sign_mnist_test_knn,
                           sign_mnist_train_knn_labels,k=3)

confusionMatrix(sign_classifier_knn_3,sign_mnist_test$label)

#model3 [acc-0.8063 ,Kappa : 0.7973 ]
sign_classifier_knn_5 <- knn(train = sign_mnist_train_knn ,test = sign_mnist_test_knn,
                           sign_mnist_train_knn_labels,k=5)

confusionMatrix(sign_classifier_knn_5,sign_mnist_test$label)

#model4 [acc- 0.796 , Kappa : 0.7866 ]
sign_classifier_knn_10 <- knn(train = sign_mnist_train_knn ,test = sign_mnist_test_knn,
                              sign_mnist_train_knn_labels,k=10)

confusionMatrix(sign_classifier_knn_10,sign_mnist_test$label)

#model5 [acc-0.6088,Kappa : 0.5909]
#Z-score standardization
sign_mnist_train_knn_z <- as.data.frame(scale(sign_mnist_train_knn))
sign_mnist_test_knn_z <- as.data.frame(scale(sign_mnist_test_knn))

sign_classifier_knn_z <- knn(train = sign_mnist_train_knn_z,test = sign_mnist_test_knn_z
                             ,sign_mnist_train_knn_labels,k=5)
confusionMatrix(sign_classifier_knn_z,sign_mnist_test$label)

#Final eval model-5
#accuracy
table(sign_classifier_knn_z,sign_mnist_test$label)
agreement <- sign_classifier_knn_z==sign_mnist_test$label
table(agreement)
prop.table(table(agreement))

#kappa statistic
library(vcd)
Kappa(table(sign_classifier_knn_z,sign_mnist_test$label))



