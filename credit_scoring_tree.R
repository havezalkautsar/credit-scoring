options(scipen = 999)
data <- read.csv("D:/Big Data for Manager/CleanCreditScoring.csv", na.strings=c(""," ","NA")); View(data)

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

# Decision Tree
# buat model decision tree tradisional
model.tree <- tree(Status ~., data = train_balance)

# plot model tree yang telah dibuat
plot(model.tree)
text(model.tree)

# Uji model tree yang dibuat pada data testing
pred.tree <- predict(model.tree, data_testing, type="class")

# confusion matrix dari hasil prediksi
cm.tree <- confusionMatrix(pred.tree,data_testing$Status, positive = "good")
cm.tree

# Conditional Inference Tree Model
# buat model conditional inference tree
model.ctree <- ctree(Status ~., data = train_balance)

# plot model tree yang telah dibuat
plot(model.ctree, type = "simple")
text(model.ctree)

# Uji model tree yang dibuat pada data testing
pred.ctree <- predict(model.ctree, newdata = data_testing, type="response")
prob.ctree <- predict(model.ctree, newdata = data_testing, type="prob")

# confusion matrix dari hasil prediksi
cm.ctree <- confusionMatrix(pred.ctree,data_testing$Status, positive = "good")
cm.ctree

# Recursive Partitioning Tree Model
# buat model rpart tree
model.rpart <- rpart(as.factor(Status) ~., data = train_balance, method="class")

# plot model tree yang telah dibuat
par(xpd=TRUE)
plot(model.rpart, compress=TRUE)
text(model.rpart)

# Uji model tree yang dibuat pada data testing
pred.rpart <- predict(model.rpart, data_testing, type="class")
prob.rpart <- predict(model.rpart, data_testing, type="prob")

# confusion matrix dari hasil prediksi
cm.rpart <- confusionMatrix(pred.rpart,data_testing$Status, positive = "good")
cm.rpart

# Random Forest Model
# buat model randomForest tree
model.randomForest <- randomForest(Status ~., data = train_balance, importance=TRUE)

# Varimplot
varImpPlot(model.randomForest, main = "Variable Importance")

# Uji model tree yang dibuat pada data testing
pred.randomForest <- predict(model.randomForest, data_testing, type = "class")
prob.randomForest <- predict(model.randomForest, data_testing, type = "prob")

# confusion matrix dari hasil prediksi
cm.randomForest <- confusionMatrix(pred.randomForest,data_testing$Status, positive = "good")

cm.tree
cm.ctree
cm.rpart
cm.randomForest