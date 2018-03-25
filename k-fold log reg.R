dataset_cv <- train_balance[sample(nrow(train_balance)),]

# Create 10 equally size folds
folds <- cut(seq(1,nrow(dataset_cv)),breaks=10,labels=FALSE)

# dataframe evaluasi
evaluate_log = data.frame(n_feature = integer(),
                         k = integer(),tp = integer(),
                         fp = integer(),fn = integer(), tn = integer())
require(party)
require(caret)

# Perform 10 fold cross validation
for(i in 1:10){
  
  set.seed(9)
  
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- dataset_cv[testIndexes, ]
  trainData <- dataset_cv[-testIndexes, ]
  #Use the test and train data partitions however you desire...
  
  model.log <- glm(Status~., data = train_balance, family = binomial())
  pred.log <- predict(model.log.step, type='response', data_testing)
  confM.log = confusionMatrix(as.numeric(pred.log > 0.5), data_testing$Status, positive = "1")
  
  # image_file = paste("log",n,"feature.png")
  # png(filename=image_file, width = 1280, height = 1280)
  # plot(models.log)
  # dev.off()
  #summary_array[n] = summary(models.log)
  temp_log = data.frame(i,confM.log$table[1],confM.log$table[2],
                                 confM.log$table[3],confM.log$table[4])
  names(temp_log) = c("k","tp","fp","fn","tn")
  evaluate_log = rbind(evaluate_log,temp_log)
}

summary_evaluate_log = evaluate_log %>% mutate(accuracy = (tp+tn)/(tp+fp+fn+tn),
                                             precision = tp/(tp+fp),
                                             recall = tp/(tp+fn),
                                             specificity = tn/(tn+fp),
                                             fpr = fp/(fp+tn))


summary_evaluation = summary_evaluate_log %>% 
  summarise(accuracy = mean(accuracy),
            precision = mean(precision),
            recall = mean(recall),
            specificity = mean(specificity),
            fpr = mean(fpr))

View(summary_evaluate_log)
View(summary_evaluation)

write.csv(summary_evaluate_log, "evaluasi 10 fold log reg.csv", row.names = FALSE)

summ <- summary_evaluate_log[,6:10]
View(summ)

# apply(as.matrix(summ), 1, function(x){mean(x)+qt(0.975,df=nrow(summ)-1)*(sd(x)/sqrt(n))})
# 
# mean(x)+qt(0.975,df=nrow(summary_evaluate_log)-1)*(sd(x)/sqrt(n))

a <- mean(summ$accuracy)
s <- sd(summ$accuracy)
n <- 10
error <- qt(c(.99, .975),df=n-1)*s/sqrt(n)
left <- a-error
right <- a+error
left; right
error

a <- mean(summ$precision)
s <- sd(summ$precision)
n <- 10
error <- qt(c(.99, .975),df=n-1)*s/sqrt(n)
left <- a-error
right <- a+error
left; right
error

a <- mean(summ$recall)
s <- sd(summ$recall)
n <- 10
error <- qt(c(.99, .975),df=n-1)*s/sqrt(n)
left <- a-error
right <- a+error
left; right
error

a <- mean(summ$specificity)
s <- sd(summ$specificity)
n <- 10
error <- qt(c(.99, .975),df=n-1)*s/sqrt(n)
left <- a-error
right <- a+error
left; right
error

a <- mean(summ$fpr)
s <- sd(summ$fpr)
n <- 10
error <- qt(c(.99, .975),df=n-1)*s/sqrt(n)
left <- a-error
right <- a+error
left; right
error