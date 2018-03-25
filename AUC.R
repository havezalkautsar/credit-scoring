# ROCR
require(ROCR)
ctree.roc <- prediction(prediction = sapply(prob.ctree, "[[", 2), data_testing$Status)
rpart.roc <- prediction(prediction = prob.rpart[,2], data_testing$Status)
randomForest.roc <- prediction(prediction = prob.randomForest, data_testing$Status)

ROCR.perf <- ROCR::performance(randomForest.roc, "tpr", "fpr")

# AUC
# AUC.perf <- ROCR::performance(randomForest.roc, "auc")
AUC.performance <- ROCR::performance(randomForest.roc, "auc")
auc.val <- as.numeric(AUC.performance@y.values); auc.val
plot(ROCR.perf, col=2, main=paste0(' Area Under the Curve = ',round(auc.val*100,3),'%'))
abline(a = 0, b = 1)

# Gini Ratio
gini.ratio <- (2*auc.val)-1; gini.ratio

# Kolmogorv Smirnov by manual
rocr.y <- attr(ROCR.perf,'y.values')[[1]]
rocr.x <- attr(ROCR.perf,'x.values')[[1]]
alpha.ks = 0.05
n.rocr.y <- length(rocr.y)
n.rocr.x <- length(rocr.x)
coef.alpha.ks = sqrt((-0.5)*(log(alpha.ks/2)))
p.val.alpha = coef.alpha.ks*sqrt((n.rocr.y+n.rocr.x)/(n.rocr.y*n.rocr.x))

h0 <- print("Label good dan bad memiliki distribusi yang sama.")
h1 <- print("Label good dan bad \n tidak memiliki distribusi yang sama.")

group <- c(rep("rocr.y", length(rocr.y)), rep("rocr.x", length(rocr.x)))
dat <- data.frame(KSD = c(rocr.y,rocr.x), group = group)
View(dat)

cdf.y <- ecdf(rocr.y)
cdf.x <- ecdf(rocr.x)

minMax <- seq(min(rocr.y, rocr.x), max(rocr.y, rocr.x), length.out=length(rocr.y)) 
x0 <- minMax[which( abs(cdf.y(minMax) - cdf.x(minMax)) == max(abs(cdf.y(minMax) - cdf.x(minMax))) )]
xn <- max(x0)

ks = max(abs(cdf.y(minMax) - cdf.x(minMax))); ks #ks manual

conclude <- ifelse(ks <= p.val.alpha,print(c("KS Test <= critical value" = h0)),print(c("KS Test > critical value" = h1)))

y0 <- cdf.y(xn)
y1 <- cdf.x(xn)

plot(cdf.y, verticals=TRUE, do.points=FALSE, col="blue", main=paste0(conclude))
plot(cdf.x, verticals=TRUE, do.points=FALSE, col="green", add=TRUE)
points(c(xn, xn), c(y0, y1), pch=16, col="red")
segments(xn, y0, xn, y1, col="red", lty="dotted")
text(size = 2,locator(), labels = paste0(' KS Test = ',round(ks*100,2),'%'))