
# spark library
# library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

# spark connection
# sc <- sparkR.init("local", "SparkR", Sys.getenv("SPARK_HOME"))

# detach sparkr
# detach("package:SparkR", unload = TRUE)

library(devtools)
devtools::install_github("dalpozz/unbalanced")
install.packages("RCurl")
install.packages("tree")
install.packages("party")
install.packages("rpart")
install.packages("randomForest")
install.packages("caret")
install.packages("nnet")
install.packages("NeuralNetTools")
install.packages("e1071")
install.packages("kernlab")
install.packages("neuralnet")
install.packages("kernlab")
install.packages("neuralnet")

library(RCurl)
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
library(nnet)
library(NeuralNetTools)

options(scipen = 999)
data <- read.csv("CleanCreditScoring.csv", na.strings=c(""," ","NA")); data

dataset <- data
str(dataset)

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

n = nrow(dataset)
set.seed(9)
idx = sample(n,n*0.7)
data_training = dataset[idx,]
data_testing = dataset[-idx,]

# proporsi Status pada kedua data
prop.table(table(data_training$Status))
prop.table(table(data_testing$Status))

# melakukan balancing data kolektibilitas agar proporsi kolektibilitas 1 dan 0 hampir sama besar
training_balanced = ubSMOTE(select(data_training, -Status),
                            data_training$Status)

# gabungkan untuk menjadi data_training
training_balance = cbind(training_balanced$X,Status = training_balanced$Y)

# proses balancing cukup lama, duplikasi ke dataframe baru agar tidak mengulang proses balancing.
train_balance <- training_balance

# Jika perlu save ke bentuk csv.
write.csv(train_balance, "data_training_balanced.csv", row.names = FALSE)

# rubah class dari biner ke good and bad
train_balance$Status = ifelse(train_balance$Status == 1, "good", "bad")
train_balance$Status <- as.factor(train_balance$Status)

data_testing$Status = ifelse(data_testing$Status == 1, "good", "bad")
data_testing$Status <- as.factor(data_testing$Status)

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

# model ANN
model.ann <- nnet(Status~., data = train_balance, size = 20)

#import function from Github for plotting nnet
require(RCurl)
 
root.url<-'https://gist.githubusercontent.com/fawda123'
raw.fun<-paste(
  root.url,
  '5086859/raw/cc1544804d5027d82b70e74b83b3941cd2184354/nnet_plot_fun.r',
  sep='/'
  )
script<-getURL(raw.fun, ssl.verifypeer = FALSE)
eval(parse(text = script))
rm('script','raw.fun')

# nnet plot
par(mar=numeric(4),mfrow=c(1,2),family='serif')
plot(model.ann,nid=F)
plot(model.ann)

# predict with ann model
pred.ann <- predict(model.ann,newdata=data_testing, type="class")
prob.ann <- predict(model.ann,newdata=data_testing, type="raw")

# confusion matrix
cm.ann <- confusionMatrix(as.factor(pred.ann), data_testing$Status, positive = "good")
cm.ann

# ROC performance
ann.pred <- ROCR::prediction(prob.ann, data_testing$Status)
ROCR.perf <- ROCR::performance(ann.pred,"tpr","fpr")

# AUC value & Gini
auc <- round(ROCR::performance(ann.pred, measure = "auc")@y.values[[1]]*100, 2)
gini <- (2*auc - 100)
cat("AUROC: ",auc,"\tGini:", gini, "\n")

# ROC plot
plot(ROCR.perf, lwd=2, colorize=TRUE,
     main=c(paste0(' Area Under the Curve = ',round(auc,3),'%'),
            paste0(' Gini Coefficient = ',round(gini,3),'%')))
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3)

# KS Alpha Coefficient & KS p-value
rocr.y <- attr(ROCR.perf,'y.values')[[1]]
rocr.x <- attr(ROCR.perf,'x.values')[[1]]
alpha.ks = 0.05
n.rocr.y <- length(rocr.y)
n.rocr.x <- length(rocr.x)
coef.alpha.ks = sqrt((-0.5)*(log(alpha.ks/2)))
p.val.alpha = coef.alpha.ks*sqrt((n.rocr.y+n.rocr.x)/(n.rocr.y*n.rocr.x))
cat("KS Alpha Coefficient: ",coef.alpha.ks,"\tKS p-value: ", p.val.alpha, "\n")

# set null hypothesis and alternate hypothesis
h0 <- print("Label good dan bad memiliki distribusi yang sama.")
h1 <- print("Label good dan bad tidak memiliki distribusi yang sama.")

# write as dataframe
group <- c(rep("rocr.y", length(rocr.y)), rep("rocr.x", length(rocr.x)))
dat <- data.frame(KSD = c(rocr.y,rocr.x), group = group)
dat

# Maximum distance of two distribution
cdf.y <- ecdf(rocr.y) # empirical distribution function of tpr
cdf.x <- ecdf(rocr.x) # empirical distribution function of fpr

# calculate xn as base point of maximum distance
minMax <- seq(min(rocr.y, rocr.x), max(rocr.y, rocr.x), length.out=length(rocr.y)) 
x0 <- minMax[which( abs(cdf.y(minMax) - cdf.x(minMax)) == max(abs(cdf.y(minMax) - cdf.x(minMax))) )]
xn <- max(x0)

# KS Test Value
ks = max(abs(cdf.y(minMax) - cdf.x(minMax)))
cat("KS Alpha Coefficient: ",coef.alpha.ks,"\tKS p-value: ", p.val.alpha,"\tKS-Test: ", ks, "\n")

# conclude
conclude <- ifelse(ks <= p.val.alpha,print(c("KS Test <= critical value" = h0)),print(c("KS Test > critical value" = h1)))

# apply to empirical distribution function of tpr & fpr
y0 <- cdf.y(xn)
y1 <- cdf.x(xn)

# plot the result
plot(cdf.y, verticals=TRUE,do.points=FALSE, col="blue", 
     main= c(paste0(' KS Test = ',round(ks*100,3),'%'),
             paste0(' Critical Value = ',round(p.val.alpha*100,3),'%'),
             paste0(conclude)))
plot(cdf.x, verticals=TRUE, do.points=FALSE, col="green", add=TRUE)
points(c(xn, xn), c(y0, y1), pch=16, col="red")
segments(xn, y0, xn, y1, col="red", lty="dotted")
