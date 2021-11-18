library(caret)
library(e1071)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(nnet)
library(ranger)

set.seed(42)

#Loading in CSV (Hardset location values)
csv <- "C:\\Users\\Randi\\Desktop\\School\\Masters\\CYBR593B\\50000passwords_tfidf_char.csv"
base <- read.csv(csv, sep=",", header=TRUE, stringsAsFactors=TRUE)

#Creating test/train data for good/bad
trainIndex <- createDataPartition(base$Label, p=0.70, list=FALSE)
train <- base[trainIndex,]
test <- base[-trainIndex,]

#Base train control
trainctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

#Logistic Regression
glm.model <- glm(Label~., data=train, family=binomial)
summary(glm.model)
glm.prob <- predict(glm.model, test, type="response")
glm.pred <- ifelse(glm.prob > 0.5, "good", "bad")
#confusionMatrix
confmatrix <- table(test$Label, glm.pred)
confmatrix
#Accuracy
(sum(diag(confmatrix)) / sum(confmatrix)*100)
TN = confmatrix[1,1]
TP = confmatrix[2,2]
FP = confmatrix[1,2]
FN = confmatrix[2,1]
#Precision
precision <- (TP)/(TP+FP)
precision
#Recall
recall <- (TP)/(TP+FN)
recall
#F1
2*((precision*recall)/(precision+recall))
        

#Random Forest Classifier
rf.model <- train(Label~., data=train, method="ranger", 
                  trControl=trainctrl, 
                  metric="Accuracy", 
                  tuneLength=10, num.trees=100)
rf.model$times
rf.predict <- predict(rf.model, test)
confusionMatrix(rf.predict, as.factor(test$Label), mode="prec_recall")
predict(rf.model, "ADer1234&^!!!dsdsds")

#Decision Tree Model
tree.model <- rpart(Label~., data=train)
tree.plot <- rpart.plot(tree.model, compress = 1, varlen = 5, tweak = 1.2, digits = 2)

tree.predict <- predict(tree.model, test, type="class")
confusionMatrix(tree.predict, as.factor(test$Label), mode="prec_recall")

#KNN
knn.model <- train(Label~., data=train, method="knn",
                   tuneLength = 10,
                   trControl = trainctrl,
                   metric="Accuracy")
knn.model$times
knn.predict <- predict(knn.model, test)
confusionMatrix(knn.predict, as.factor(test$Label), mode="prec_recall")
ggplot(data=knn.model, aes(x=knn.model$C, y=knn.model$accuracy))

#SVM
svm.model <- train(Label~., data=train, method="svmRadial",
                   tuneLength = 10,
                   trControl = trainctrl,
                   metric="Accuracy")
svm.model$times
svm.predict <- predict(svm.model, test)
confusionMatrix(svm.predict, as.factor(test$Label), mode="prec_recall")
ggplot(data=svm.model, aes(x=svm.model$C, y=svm.model$accuracy))


#SVM GridSearch
svmGrid <- expand.grid(sigma = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), C = c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128))
svm.model2 <- train(Label~., data=train, method="svmRadial",
                    trControl = trainctrl, tuneGrid = svmGrid)
svm.model2$times
svmModel2Predict <- predict(svm.model2, test)
confusionMatrix(svmModel2Predict, as.factor(test$Label), mode="prec_recall")

#Neural Net
nnet.model <- train(Label~., data=train, method="nnet",
                    tuneLength = 10,
                    trControl = trainctrl,
                    metric="Accuracy")
nnet.model$times
nnet.predict <- predict(nnet.model, test)
confusionMatrix(nnet.predict, as.factor(test$Label), mode = "prec_recall")
ggplot(data=nnet.model, aes(x=nnet.model$size, y=nnet.model$decay))
