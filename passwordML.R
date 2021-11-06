library(caret)
library(rpart)
library(rpart.plot)
library(data.table)
library(ggplot2)
library(corrgram)
library(tokenizers)
library(superml)
library(glmnet)
library(survival)
library(lattice)
library(rms)
library(dplyr)
library(e1071)

#Hard set dataset location and load it in
passwordDataSetLoc <- "C:/Users/Randi/Desktop/School/CYBR593B/passworddataset.csv"
#passwordCSV <- read.csv(passwordDataSetLoc, sep=",")
passwordCSV <- fread(passwordDataSetLoc)

y = as.vector(passwordCSV$Label)
allPasswords = as.vector(passwordCSV$Password)

vectorizer = TfIdfVectorizer$new()
x = vectorizer$fit_transform(allPasswords) #Creates "Error: cannot allocate vector of size 3808899.5 Gb on machine with 16GB RAM
nSample = NROW(x)
w = 0.8
xTrain = x[0:(w*nSample),]
yTrain = y[0:(w*nSample)]
xTest = x[((w*nSample)+1):nSample,]
yTest = y[((w*nSample)+1):nSample]

#Logistic Regression
modelLambda <- cv.glmnet(as.matrix(xTrain), as.factor(yTrain), nfolds=10, alpha=1, family="binomial", type.measure="class")
pred <- as.numeric(predict(modelLambda, newx=as.matrix(xTest), type="class"))
lgSum = (yTest==pred)/NROW(pred)
sprintf("Logistic regression prediction score: %d", lgSum)

#===========================================
#OLD TESTS
#===========================================
#Double check loaded correctly and there are no NA values (any should give FALSE reading)
str(passwordCSV)
any(is.na(passwordCSV))

#Creating tokens via character tokenizatin
tokens <- tokenize_characters(passwordCSV$Password, strip_non_alphanum=FALSE, lowercase=FALSE, simplify=FALSE)

#Quick plot to see correlation between attributes
corrgram(tokens, lower.panel=panel.shade, upper.panel=panel.cor)

#Set seed
set.seed(69)

#Create partition, train, and test sets
trainIndex <- createDataPartition(passwordCSV$Label, p=0.70, list=FALSE)
Train <- passwordCSV[trainIndex,]
Test <- passwordCSV[-trainIndex,]

#Create Linear Regression Model
lrModel <- lm(formula=Length~., data=Train)
summary(lrModel)

#Create residual histogram
rHisto <- as.data.frame(residuals(lrModel))
ggplot(rHisto, aes(residuals(lrModel))) + geom_histogram(fill="red", color="gray")

prediction <- predict(lrModel, Test)

treeModel <- rpart(Label~., data=Train)
treePlot <- rpart.plot(treeModel)
treePlot <- rpart.plot(treeModel, compress = 1, varlen = 5, tweak = 1.2, digits = 2)

Prediction <- predict(treeModel, Test, type="class")  
Prediction
