library(tidyverse)
library(dplyr)
library(ggplot2)
library(stringr)
library(scales)

# Load data

data <- read.csv("D:/Big Data for Manager/CleanCreditScoring.csv")
req <- data
options(scipen = 999)
req <- na.omit(req)
request <- req

View(request)
str(request)

request <- request[,1:16]
str(request)

nums <- sapply(request, is.numeric)
request[ ,nums] <- lapply(request[ ,nums], as.numeric)

str(request)

table(request$Home, request$Status)

colSums(is.na(request))

# contigency table
df.Home <- request %>%
  group_by(Home, Status) %>%
  summarise(counts = n())

# distribution plot
ggplot(df.Home, aes(x = Home, y = counts)) +
  geom_bar(aes(color = Status, fill = Status), stat = "identity",
           position = position_dodge(0.8), width = 0.7) +
  geom_text(aes(label = counts, group = Status), position = position_dodge(0.8),
            vjust = -0.3, size = 3.5) +
  scale_x_discrete(labels = wrap_format(10))

# chi-square test
tabel <- xtabs(counts ~ Home + Status, df.Home)
tabel
chi <- chisq.test(tabel)
chi; chi$statistic; chi$parameter; chi$p.value
formatC(chi$p.value, format = "e", digits = 2)
chi$observed
chi$expected
chi$residuals
chi$stdres
chidf <- as.data.frame(cbind(chi$observed, chi$expected, chi$residuals, chi$stdres))