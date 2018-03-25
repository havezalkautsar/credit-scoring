library(kernlab)
library(neuralnet)
library(dplyr)
library(unbalanced)
library(tidyverse)
library(tree)
library(party)
library(rpart)
library(randomForest)
library(caret)
library(e1071)


options(scipen = 999)
data <- read.csv("D:/Big Data for Manager/CleanCreditScoring.csv", na.strings=c(""," ","NA"))
View(data)

dataset <- data
str(dataset)

# Data Preprocessing

# Buang kolom-kolom identitas, kolom yang terlalu banyak NA, kolom yang redundan, kolom yang bernilai sama disemua baris, dan kolom yang sudah pasti menandakan kondisi credit.
# drop redundant range data
dataset <- dataset[,1:16]
str(dataset)

# rubah class dari biner ke good and bad
dataset$Status = ifelse(dataset$Status == "good", 1, 0)
dataset$Status <- as.factor(dataset$Status)

# rubah class dari no_rec & yes_rec ke no & yes
dataset$Records = ifelse(dataset$Records == "no_rec", "no", "yes")
dataset$Records <- as.factor(dataset$Records)

str(dataset)

prop.table(table(dataset$Status))

# bagi menjadi 70% data training dan 30% data secara random
n = nrow(dataset)
set.seed(9)
idx = sample(n,n*0.7)
data_training = dataset[idx,]
data_testing = dataset[-idx,]

# proporsi Status pada kedua data
prop.table(table(data_training$Status))
prop.table(table(data_testing$Status))

#Proporsi credit status di kedua data sangat tidak balance, oleh karena itu pada data training perlu dilakukan balancing agar proporsi good dan bad menjadi seimbang.
# melakukan balancing data kolektibilitas agar proporsi kolektibilitas 1 dan 0 hampir sama besar
training_balanced = ubSMOTE(select(data_training, -Status),
                            data_training$Status)

# gabungkan untuk menjadi data_training
training_balance = cbind(training_balanced$X,Status = training_balanced$Y)

# proses balancing cukup lama, duplikasi ke dataframe baru agar tidak mengulang proses balancing.
train_balance <- training_balance

# Jika perlu save ke bentuk csv.
write.csv(train_balance, "data_training_balanced.csv", row.names = FALSE)

# # rubah class dari biner ke good and bad
# train_balance$Status = ifelse(train_balance$Status == 1, "good", "bad")
# train_balance$Status <- as.factor(train_balance$Status)
# 
# data_testing$Status = ifelse(data_testing$Status == 1, "good", "bad")
# data_testing$Status <- as.factor(data_testing$Status)

# proporsi credit_status pada kedua data
prop.table(table(train_balance$Status))
table(train_balance$Status)

prop.table(table(data_testing$Status))
table(data_testing$Status)

# struktur kedua data
str(train_balance)
str(data_testing)

# pindah posisi Status pada data_testing
data_testing <- data_testing[,c(2:16,1)]

# hitung NA
colSums(is.na(train_balance))
colSums(is.na(data_testing))

# drop NA 
# train_balance <- na.omit(train_balance)
# data_testing <- na.omit(data_testing)

# cari perbedaan kedua data
inner_join(train_balance, data_testing)

# class kedua data
sapply(train_balance, class)
sapply(data_testing, class)

# samakan class kedua data
nums <- sapply(train_balance, is.numeric)
train_balance[ ,nums] <- lapply(train_balance[ ,nums], as.numeric)

num <- sapply(data_testing, is.numeric)
data_testing[ ,num] <- lapply(data_testing[ ,num], as.numeric)

# class kedua data
sapply(train_balance, class)
sapply(data_testing, class)

# modelling logistic regression
model.log <- glm(Status~., data = train_balance, family = binomial())
model.log.step <- step(model.log)
summary(model.log.step)

# List of significant variables and features with p-value <0.01
significant.variables <- summary(model.log.step)$coeff[-1,4] < 0.05
names(significant.variables)[significant.variables == TRUE]

# predict
pred <- predict(model.log.step, type = "response")
res <- residuals(model.log.step, type = "deviance")

#Plot Residuals
plot(predict(model.log.step), res,
     xlab="Fitted values", ylab = "Residuals",
     ylim = max(abs(res)) * c(-1,1))

# CIs using profiled log-likelihood
confint(model.log.step)

# CIs using standard errors
confint.default(model.log.step)

# odds ratios and 95% CI
exp(cbind(OR = coef(model.log.step), confint(model.log.step)))

#score test data set
pred.log <- predict(model.log.step, type='response', data_testing)

# confusion matrix dari hasil prediksi
cm.log <- confusionMatrix(as.numeric(pred.log > 0.5), data_testing$Status, positive = "1")
cm.log

#score test data set
log.pred <- ROCR::prediction(pred.log, data_testing$Status)
ROCR.perf <- ROCR::performance(log.pred,"tpr","fpr")

# KS, Gini & AUC model
ks <- round(max(attr(ROCR.perf,'y.values')[[1]]-attr(ROCR.perf,'x.values')[[1]])*100, 2)
auc <- round(ROCR::performance(log.pred, measure = "auc")@y.values[[1]]*100, 2)
gini <- (2*auc - 100)
cat("AUROC: ",auc,"\tKS: ", ks, "\tGini:", gini, "\n")

# ROC plot
plot(ROCR.perf, lwd=2, colorize=TRUE,
     main=c(paste0(' Area Under the Curve = ',round(auc,3),'%'),
            paste0(' Gini Coefficient = ',round(gini,3),'%')))
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3)

# Kolmogorv Smirnov plot
rocr.y <- attr(ROCR.perf,'y.values')[[1]]
rocr.x <- attr(ROCR.perf,'x.values')[[1]]
alpha.ks = 0.05
n.rocr.y <- length(rocr.y)
n.rocr.x <- length(rocr.x)
coef.alpha.ks = sqrt((-0.5)*(log(alpha.ks/2)))
p.val.alpha = coef.alpha.ks*sqrt((n.rocr.y+n.rocr.x)/(n.rocr.y*n.rocr.x))

h0 <- print("Label good dan bad memiliki distribusi yang sama.")
h1 <- print("Label good dan bad tidak memiliki distribusi yang sama.")

group <- c(rep("rocr.y", length(rocr.y)), rep("rocr.x", length(rocr.x)))
dat <- data.frame(KSD = c(rocr.y,rocr.x), group = group)
View(dat)

cdf.y <- ecdf(rocr.y)
cdf.x <- ecdf(rocr.x)

minMax <- seq(min(rocr.y, rocr.x), max(rocr.y, rocr.x), length.out=length(rocr.y)) 
x0 <- minMax[which( abs(cdf.y(minMax) - cdf.x(minMax)) == max(abs(cdf.y(minMax) - cdf.x(minMax))) )]
xn <- max(x0)

conclude <- ifelse(ks <= p.val.alpha,print(c("KS Test <= critical value" = h0)),print(c("KS Test > critical value" = h1)))

y0 <- cdf.y(xn)
y1 <- cdf.x(xn)

plot(cdf.y, verticals=TRUE,do.points=FALSE, col="blue", 
     main= c(paste0(' KS Test = ',round(ks,3),'%'),
             paste0(' Critical Value = ',round(p.val.alpha*100,3),'%'),
             paste0(conclude)))
plot(cdf.x, verticals=TRUE, do.points=FALSE, col="green", add=TRUE)
points(c(xn, xn), c(y0, y1), pch=16, col="red")
segments(xn, y0, xn, y1, col="red", lty="dotted")
text(size = 2,locator(), labels = paste0(' KS Test = ',round(ks,2),'%'))
