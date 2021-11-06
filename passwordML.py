from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import sys
import os
import math

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

#Uses a list of passwords to verify the ML algorithm works
def predictMe(modelType, model, vectorizor):
    headers = ["Password", "Label"]
    predict = ["ADer1234&^!!!dsdsds", "hello", "wakawaka", "fdsafdadfsadfsdfa", "fdsfdsadsfA", "gtrdeA@!"]
    xPredict = vectorizer.transform(predict)
    yPredict = model.predict(xPredict)

    print("\tPrediction of passwords")
    num = 0
    printMe = []
    for password in predict:
        printMe.append([password, yPredict[num]])
        num += 1

    print("\t{}".format(pd.DataFrame(headers, printMe)))
    print("\n")

#Results of the ML algorithms
def reviewResults(modelType, model, fitted, xTrain, yTrain, xTest, yTest, vectorizer):
    print(
    """******************************
RESULTS for {} 
******************************""".format(modelType))

    ret = fitted.predict(xTest)

    print("\tAccuracy Score: {}".format(accuracy_score(yTest, ret)))
    print("\tPrecision Score: {}".format(precision_score(yTest, ret, average="binary", pos_label="bad")))
    print("\tRecall Score: {}".format(recall_score(yTest, ret, average="binary", pos_label="bad")))
    print("\tF1 Score: {}".format(f1_score(yTest, ret, average="binary", pos_label="bad")))

    predictMe(modelType, model, vectorizer)

#Logistic Regression
def LogR(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing Logistic Regression\n")
    lgs = LogisticRegression()
    fit = lgs.fit(xTrain, yTrain)
    reviewResults("Logistic Regression", lgs, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Custom function to separate each password into it's individual characters
def getTokens(input):
    #split all characters
    tokens = []
    for i in input:
        tokens.append(i)
    return tokens

#Creates the xTrain, yTrain, xTest, yTest data for the models
def createTrainTestData(path):
    print("Creating data set\n")
    passwordsCSV = pd.read_csv(path, ',', on_bad_lines="skip", engine="python")

    yLabels = passwordsCSV["Label"]
    allPass = passwordsCSV["Password"].values.astype('U')

    vectorizer = TfidfVectorizer(tokenizer=getTokens, lowercase=False)
    x = vectorizer.fit_transform(allPass)

    xTrain, xTest, yTrain, yTest = train_test_split(x, yLabels, test_size=0.3, random_state=42)

    return xTrain, yTrain, xTest, yTest, vectorizer
    
if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "passworddataset.csv")

    xTrain, yTrain, xTest, yTest, vectorizer = createTrainTestData(path)

    LogR(xTrain, yTrain, xTest, yTest, vectorizer)
    

    
