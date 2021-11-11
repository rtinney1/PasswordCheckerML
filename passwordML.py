"""
File: passwordML.py
Creator: Randi Tinney
Date Created: 31 Oct 2021
Uses various machine learning algorithms to determine if passwords are "good" or "bad".
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, explained_variance_score, max_error, r2_score, mean_gamma_deviance
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np

import argparse
import random
import sys
import os
import math
import time

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

    #not all algorithms can give the following scores and metrics. Wrapped each in individual 
    #   try/excepts so all those that can be reported, will be reported, without causing an 
    #   error to occur and crash the program
    try:
        print("\tAccuracy Score: {}".format(accuracy_score(yTest, ret)))
    except:
        pass 
    try:
        print("\tPrecision Score: {}".format(precision_score(yTest, ret, average="binary", pos_label="bad")))
    except:
        pass 
    try:    
        print("\tRecall Score: {}".format(recall_score(yTest, ret, average="binary", pos_label="bad")))
    except:
        pass
    try:
        print("\tF1 Score: {}".format(f1_score(yTest, ret, average="binary", pos_label="bad")))
    except:
        pass
    try:
        print("\tExplained Variance Score: {}".format(explained_variance_score(yTest, ret)))
    except:
        pass
    try:
        print("\tMax Error: {}".format(max_error(yTest, ret)))
    except:
        pass
    try:
        print("\tR2 Score: {}".format(r2_score(yTest, ret)))
    except:
        pass 
    try:
        print("\tMean Gamma Deviance: {}".format(mean_gamma_deviance(yTest, ret)))
    except:
        pass

    predictMe(modelType, model, vectorizer)

#Supprt Vecter Classifier ML Algorithm
def SVMC(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing SVC\n")
    svmc = SVC()
    fit = svmc.fit(xTrain, yTrain)
    reviewResults("SVC", svmc, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Support Vector Regression ML algorithm
def SVMR(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing SVR\n")
    svmr = SVR()
    conYTrain = []
    #Needed for Support Vector Regression
    print("Converting 'good' values to 1 and 'bad' values to 0. This means that the closer the value is to 1, the better the password.\n")
    for x in yTrain:
        if x == "good":
            conYTrain.append(1)
        else:
            conYTrain.append(0)
    conYTest = []
    for x in yTest:
        if x == "good":
            conYTest.append(1)
        else:
            conYTest.append(0)
    fit = svmr.fit(xTrain, conYTrain)
    reviewResults("SVR", svmr, fit, xTrain, conYTrain, xTest, conYTest, vectorizer)

#K-Nearest Neighbor
def KNearN(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing K-Nearest Neighbor\n")
    knn = KNeighborsClassifier()
    fit = knn.fit(xTrain, yTrain)
    reviewResults("K-Nearest Neighbor", knn, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Linear Regression
def LinR(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing Linear Regression\n")
    linr = LinearRegression()
    conYTrain = []
    #Needed for Linear Regression
    print("Converting 'good' values to 1 and 'bad' values to 0. This means that the closer the value is to 1, the better the password.\n")
    for x in yTrain:
        if x == "good":
            conYTrain.append(1)
        else:
            conYTrain.append(0)
    conYTest = []
    for x in yTest:
        if x == "good":
            conYTest.append(1)
        else:
            conYTest.append(0)
    fit = linr.fit(xTrain, conYTrain)
    reviewResults("Linear Regression", linr, fit, xTrain, conYTrain, xTest, conYTest, vectorizer)

#Random Forest Classifier ML algorithm
def RanF(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing Random Forest Classifier\n")
    rfc = RandomForestClassifier()
    fit = rfc.fit(xTrain, yTrain)
    reviewResults("Random Forest Classifier", rfc, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Decision Tree Classifier
def DecT(xTrain, yTrain, xTest, yTest, vectorizer):
    print("Performing Decision Tree Classifier\n")
    dfc = DecisionTreeClassifier()
    fit = dfc.fit(xTrain, yTrain)
    reviewResults("Decision Tree Classifier", dfc, fit, xTrain, yTrain, xTest, yTest, vectorizer)

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
    print("Creating testing and training data\n")
    passwordsCSV = pd.read_csv(path, ',', on_bad_lines="skip", engine="python")

    yLabels = passwordsCSV["Label"]
    allPass = passwordsCSV["Password"].values.astype('U')

    vectorizer = TfidfVectorizer(tokenizer=getTokens, lowercase=False)
    x = vectorizer.fit_transform(allPass)

    xTrain, xTest, yTrain, yTest = train_test_split(x, yLabels, test_size=0.3, random_state=42)

    return xTrain, yTrain, xTest, yTest, vectorizer

def forceOptions(value):
    val = int(value)
    if val >= 0 and val < 8:
        return val 
    raise argparse.ArgumentTypeError("\nERROR: {} is an invalid option".format(val))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Use passworddataset.csv with machine learning algorithms to determine whether passwords are good or bad")
    parser.add_argument("-m", "--ml", help="""Machine Learning algorithm to use
1\tLogistic Regression
2\tRandom Forest Classifier
3\tDecision Tree Classifier
4\tLinear Regression
5\tK-Nearest Neighbors
6\tSupport Vector Regression
7\tSupport Vector Classifier
""", required=True, type=forceOptions)
    parser.add_argument("-f", "--file", help="Dataset to use")
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), args.file)

    if not os.path.exists(path):
        print("Couldn't find {}".format(args.file))
        sys.exit()

    xTrain, yTrain, xTest, yTest, vectorizer = createTrainTestData(path)

    if args.ml == 1 or args.ml == 0:
        #Logistic Regression
        start = time.time()
        LogR(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
    if args.ml == 2 or args.ml == 0:
        #Random Forest Classifier
        start = time.time()
        RanF(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
    if args.ml == 3 or args.ml == 0:
        #Decision Tree Classifier
        start = time.time()
        DecT(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
    if args.ml == 4 or args.ml == 0:
        #Linear Regression
        start = time.time()
        LinR(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
    if args.ml == 5 or args.ml == 0:
        #12+ hours running on predict. Need more RAM/faster GPU
        #K-Nearest Neighbors
        start = time.time()
        KNearN(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
    if args.ml == 6 or args.ml == 0:
        #Suport Vector Regression
        start = time.time()
        SVMR(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
    if args.ml == 7 or args.ml == 0:
        #Support Vector Classifier
        start = time.time()
        SVMC(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        print("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        print("****************************************")
