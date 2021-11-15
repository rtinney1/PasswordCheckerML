library(caret)
library(e1071)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(nnet)

set.seed(42)

#Loading in CSV (Hardset location values)
csvChar <- "FILE LOCATION OF tfidf_char.csv FILE CREATED FROM createPasswordDataSet.py"
csvInt <- "FILE LOCATION OF tfidf_int.csv FILE CREATED FROM createPasswordDataSet.py"
baseChar <- read.csv(csvChar, sep=",")
baseInt <- read.csv(csvInt, sep=",")

#Creating test/train data for good/bad
trainIndex <- createDataPartition(baseChar$Label, p=0.70, list=FALSE)
trainChar <- baseChar[trainIndex,]
testChar <- baseChar[-trainIndex,]

#Creating test/train data for 1/0
trainIndexInt <- createDataPartition(baseInt$Label, p=0.70, list=FALSE)
trainInt <- baseInt[trainIndexInt,]
testInt <- baseInt[-trainIndexInt,]

#Logistic Regression - Requires Int data
##  I get data but don't know how to read it
logReg <- glm(Label~., data=trainInt, family=binomial)
LogRegPredict <- predict(logReg, testInt)
logRedPredict

#Decision Tree Model
treemodel <- rpart(Label~., data=trainChar)
treeplot <- rpart.plot(treemodel)
treeplot <- rpart.plot(treemodel, compress = 1, varlen = 5, tweak = 1.2, digits = 2)

treeModelPredict <- predict(treemodel, testChar, type="class")
confusionMatrix(treeModelPredict, as.factor(testChar$Label), mode="prec_recall")

#Linear Regression - Requires int data
##  I get data but don't know
lrModel <- lm(Label~., data=trainInt)
lrPredict <- predict(lrModel, testInt)
lrPredict

#KNN
knn.model <- train(Label~., data=trainChar, method="knn",
                   tuneLength = 10,
                   trControl = trainctrl,
                   metric="Accuracy")
knn.predict <- predict(knn.model, testChar)
confusionMatrix(knn.predict, as.factor(testChar$Label), mode="prec_recall")
ggplot(data=knn.model, aes(x=knn.model$C, y=knn.model$accuracy))

#SVM
trainctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
svm.model <- train(Label~., data=trainChar, method="svmRadial",
                   tuneLength = 10,
                   trControl = trainctrl,
                   metric="Accuracy")
svm.predict <- predict(svm.model, testChar)
confusionMatrix(svm.predict, as.factor(testChar$Label), mode="prec_recall")
ggplot(data=svm.model, aes(x=svm.model$C, y=svm.model$accuracy))


#SVM GridSearch
svmGrid <- expand.grid(sigma = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), C = c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128))
svm.model2 <- train(Label~., data=trainChar, method="svmRadial",
                    trControl = trainctrl, tuneGrid = svmGrid)
svmModel2Predict <- predict(svm.model2, testChar)
confusionMatrix(svmModel2Predict, as.factor(testChar$Label), mode="prec_recall")

#Neural Net
nnet.model <- train(Label~., data=trainChar, method="nnet",
                    tuneLength = 10,
                    trControl = trainctrl,
                    metric="Accuracy")
nnet.predict <- predict(nnet.model, testChar)
confusionMatrix(nnet.predict, as.factor(testChar$Label), mode = "prec_recall")
ggplot(data=nnet.model, aes(x=nnet.model$size, y=nnet.model$decay))
