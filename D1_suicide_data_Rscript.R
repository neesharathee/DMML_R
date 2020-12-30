set.seed(1)
setwd("/Users/neesharathee/Desktop/Neesha/data2")
##Step1 - Collecting the data
#importing data
suicide_data = read.csv("suicide_data.csv",stringsAsFactors = FALSE)

##Step2 - Exploring and preparing the data
#checking structure of data
summary(suicide_data)
str(suicide_data)


#checking for NAs in data
sum(is.na(suicide_data))
library(VIM)
mp <- aggr(suicide_data, col=c('navyblue','yellow'),
           numbers=TRUE, sortVars=TRUE,
           labels=names(suicide_data), cex.axis=.7,
           gap=3, ylab=c("Missing data","Pattern"))
par(mar ="3,3,3,3")

#Fixing NAs
ImputedHDIMean <- suicide_data$HDI.for.year
meanHDI = mean(ImputedHDIMean,na.rm=T)
ImputedHDIMean = ifelse(is.na(ImputedHDIMean),meanHDI,ImputedHDIMean)
par(mfrow=c(1,2))
hist(ImputedHDIMean)
hist(suicide_data$HDI.for.year)

ImputedHDIMedian <- suicide_data$HDI.for.year
ImputedHDIMedian[is.na(ImputedHDIMedian)] <- median(ImputedHDIMedian, na.rm = TRUE) 
par(mfrow=c(1,2))
hist(suicide_data$HDI.for.year)
hist(ImputedHDIMedian)

#colored graphs
hist(suicide_data$HDI.for.year, freq=F, main='HDI: Original', col='red', ylim=c(0,0.04)) 
hist(ImputedHDIMean, freq=F, main='HDI: Imputed HDI Mean', col='blue', ylim=c(0,0.04))


library(mice)
mice_mod <- mice(suicide_data[, !names(suicide_data) %in%
                               c('country','year','sex','age','country.year','generation')], method='rf')
summary(mice_mod)
mice_output <- complete(mice_mod)
length(mice_output$HDI.for.year)
df <- data.frame(suicide_data$HDI.for.year, ImputedHDIMean, ImputedHDIMedian, mice_output$HDI.for.year)
summary(df)
sapply(df, function(x) sd(x, na.rm=T))
#mice model seems to retain mean and median so it is chosen
suicide_data$HDI.for.year <- mice_output$HDI.for.year
sum(is.na(suicide_data))

#transforming data
suicide_data <- suicide_data[,-c(1,2)]
suicide_data$country.year <- factor(suicide_data$country.year)
suicide_data$sex <- factor(suicide_data$sex)
suicide_data$age <- factor(suicide_data$age)
suicide_data$generation <- factor(suicide_data$generation)
gdpperyear <- suicide_data$gdp_for_year....
gdpperyear <- as.numeric(gsub(",", "", gdpperyear))
suicide_data$gdp_for_year.... <- gdpperyear/1000000
suicide_data$population <- suicide_data$population/100
suicide_data$HDI.for.year <- suicide_data$HDI.for.year*1000
suicide_data$risk <- ifelse(suicide_data$suicides.100k.pop>mean(suicide_data$suicides.100k.pop),1,0)
suicide_data$risk <- factor(suicide_data$risk)
table(suicide_data$risk)
names(suicide_data) <- c("sex","age","num_suicide","pop","rate_suicide","country_year","HDI","GDP","rate_GDP","gen","risk")
str(suicide_data)


#visualising risk with respect to different predictors
library(ggthemes)
ggplot(suicide_data, aes(x = sex, fill = factor(risk))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Sex') + theme_few()
ggplot(suicide_data, aes(x = age, fill = factor(risk))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Age') + theme_few()
ggplot(suicide_data, aes(x = HDI, fill = factor(risk))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'HDI') + theme_few()
ggplot(suicide_data, aes(x = gen, fill = factor(risk))) + geom_bar(stat='count',position='dodge') +
  labs(x = 'Generation') + theme_few()

#checking for normal distribution of dependent variable
hist(suicide_data$num_suicide, # histogram
     col="grey", # column color
     border="black",
     breaks = 10,
     prob = TRUE, # show densities instead of frequencies
     xlab = "residuals",
     main = "Histogram of number of suicides")
lines(density(suicide_data$num_suicide), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")
boxplot(suicide_data$num_suicide,main ="suicide data",col = '#A4A4A4' )
qqnorm(suicide_data$num_suicide, ylab="Q-Q plot of number of suicides")
qqline(suicide_data$num_suicide,col="red")
#If the skewness of the predictor variable is less than -1 or greater than +1, the data is highly skewed
library(e1071)
skewness(suicide_data$num_suicide)

#log transformation
log_num_suicide <- log1p(suicide_data$num_suicide)
summary(log_num_suicide)
suicide_data$num_suicide <- log_num_suicide

hist(suicide_data$num_suicide, # histogram
     col="grey", # column color
     border="black",
     breaks = 5,
     prob = TRUE, # show densities instead of frequencies
     xlab = "residuals",
     main = "Histogram of number of suicides")
lines(density(suicide_data$num_suicide), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")
boxplot(suicide_data$num_suicide,main ="suicide data",col = '#A4A4A4' )
qqnorm(suicide_data$num_suicide, ylab="Q-Q plot of number of suicides")
qqline(suicide_data$num_suicide,col="red")
#If the skewness of the predictor variable is -0.5 and +0.5, the data is approximately symmetric
library(e1071)
skewness(suicide_data$num_suicide)

#test and train samples
#stratified random sampling
library(caret)
index <-createDataPartition(suicide_data$risk,p=0.75,list = FALSE)
suicide_train <- suicide_data[index, ]
row.names(suicide_train) <- 1:20866
suicide_test <- suicide_data[-index, ]
row.names(suicide_test) <- 1:6954

#finding correlations between variables
library(psych)
pairs.panels(suicide_train)
cor(suicide_train[,unlist(lapply(suicide_train, is.numeric))])

##Step3 - Training a model on the data
######################### multiple linear regression
#best subset regression
library(leaps)
library(car)
BestFit.model <- regsubsets(num_suicide~pop+rate_suicide+HDI+GDP+rate_GDP+gen+sex+age,data=suicide_train,nbest=2,really.big = TRUE)
par(mfrow=c(1,1))
plot(BestFit.model,scale="adjr2")

#model1 
model1 <- lm(num_suicide~pop+rate_suicide+HDI+age+sex,data=suicide_train)
summary(model1)
par(mfrow=c(2,2))
plot(model1)
hist(model1$residuals)

# predicting test values
model1.pred <- predict(model1,suicide_test)
# a) Compute the prediction error, RMSE = 1.40
RMSE(model1.pred, suicide_test$num_suicide)
# b) Compute R-square = 0.607
R2(model1.pred, suicide_test$num_suicide)

#Simple plot of predicted values with 1-to-1 line
suicide_test$pred = model1.pred
plot(pred ~ num_suicide,
     data=suicide_test,
     pch = 16,
     xlab="Actual response value",
     ylab="Predicted response value")
abline(0,1, col="blue", lwd=2)



#model2 
model2 <- lm(num_suicide~pop+rate_suicide+HDI+rate_GDP+age+sex,data=suicide_train)
summary(model2)
par(mfrow=c(2,2))
plot(model2)
hist(model2$residuals)

# predicting test values
model2.pred <- predict(model2,suicide_test)
# a) Compute the prediction error, RMSE=1.40
RMSE(model2.pred, suicide_test$num_suicide)
# b) Compute R-square=0.608
R2(model2.pred, suicide_test$num_suicide)

#Simple plot of predicted values with 1-to-1 line
suicide_test$pred = model2.pred
plot(pred ~ num_suicide,
     data=suicide_test,
     pch = 16,
     xlab="Actual response value",
     ylab="Predicted response value")
abline(0,1, col="blue", lwd=2)

##Step5 - Improving model performance
#using numeric var to binary conversion : risk
#using interactions between population,suicide rate and  GDP rate
#model3 
model3 <- lm(num_suicide~pop+rate_suicide+HDI+rate_GDP+age+sex+risk+pop:rate_suicide+pop:rate_GDP,data=suicide_train)
summary(model3)
par(mfrow=c(2,2))
plot(model3)

# predicting test values
model3.pred <- predict(model3,suicide_test)
# a) Compute the prediction error, RMSE=1.32
RMSE(model3.pred, suicide_test$num_suicide)
# b) Compute R-square=0.65
R2(model3.pred, suicide_test$num_suicide)

#Simple plot of predicted values with 1-to-1 line
suicide_test$pred = model3.pred
plot(pred ~ num_suicide,
     data=suicide_test,
     pch = 16,
     xlab="Actual response value",
     ylab="Predicted response value")
abline(0,1, col="blue", lwd=2)

#tests for linear model
library(car)
vif(model3)
summary(cooks.distance(model3))
influencePlot(model = model3, scale =3, main = "Influence plot")
#removing outliers
suicide_train = suicide_train[-c(943,15818,15828,16737),]
row.names(suicide_train) <- 1:20862

#model4 
model4 <- lm(num_suicide~pop+rate_suicide+HDI+rate_GDP+age+sex+risk+pop:rate_suicide+pop:rate_GDP,data=suicide_train)
summary(model4)
par(mfrow=c(2,2))
plot(model4)

# predicting test values
model4.pred <- predict(model4,suicide_test)
# a) Compute the prediction error, RMSE=1.32
RMSE(model4.pred, suicide_test$num_suicide)
# b) Compute R-square=0.65
R2(model4.pred, suicide_test$num_suicide)

#Simple plot of predicted values with 1-to-1 line
suicide_test$pred = model4.pred
plot(pred ~ num_suicide,
     data=suicide_test,
     pch = 16,
     xlab="Actual response value",
     ylab="Predicted response value")
abline(0,1, col="blue", lwd=2)

influencePlot(model = model4, scale =3, main = "Influence plot")
#removing outliers
suicide_train = suicide_train[-c(15751,15779,15835,16334,18273),]
row.names(suicide_train) <- 1:20857

#model5 
model5 <- lm(num_suicide~pop+rate_suicide+HDI+rate_GDP+age+sex+risk+pop:rate_suicide+pop:rate_GDP,data=suicide_train)
summary(model5)
par(mfrow=c(2,2))
plot(model5)

# predicting test values
model5.pred <- predict(model5,suicide_test)
# a) Compute the prediction error, RMSE=1.32
RMSE(model5.pred, suicide_test$num_suicide)
# b) Compute R-square=0.65
R2(model5.pred, suicide_test$num_suicide)

#Simple plot of predicted values with 1-to-1 line
suicide_test$pred = model5.pred
plot(pred ~ num_suicide,
     data=suicide_test,
     pch = 16,
     xlab="Actual response value",
     ylab="Predicted response value")
abline(0,1, col="blue", lwd=2)

#Step 7 knowledge discovery
library(huxtable)
library(flextable)
huxreg(model1, model2, model3, model4, model5)





