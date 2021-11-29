"""
File: passwordML.py
Creator: Randi Tinney
Date Created: 31 Oct 2021
Uses various machine learning algorithms to determine if passwords are "good" or "bad".
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, explained_variance_score, max_error, r2_score, mean_gamma_deviance
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np

import argparse
import random
import sys
import os
import math
import time
import textwrap

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

outfile = "outfile.txt"

def myprint(mystring):
    print(mystring)
    with open(outfile, "a") as f:
        f.write(mystring)

#Uses a list of passwords to verify the ML algorithm works
def predictMe(modelType, model, vectorizor):
    headers = ["Password", "Label"]
    predict = ["ADer1234&^!!!dsdsds", "hello", "wakawaka", "fdsafdadfsadfsdfa", "fdsfdsadsfA", "gtrdeA@!"]
    xPredict = vectorizer.transform(predict)
    yPredict = model.predict(xPredict)

    myprint("\tPrediction of passwords")
    num = 0
    printMe = []
    for password in predict:
        printMe.append([password, yPredict[num]])
        num += 1

    myprint("\t{}".format(pd.DataFrame(headers, printMe)))
    myprint("\n")

#Results of the ML algorithms
def reviewResults(modelType, model, fitted, xTrain, yTrain, xTest, yTest, vectorizer):
    myprint(
    """******************************
RESULTS for {} 
******************************""".format(modelType))

    ret = fitted.predict(xTest)

    #not all algorithms can give the following scores and metrics. Wrapped each in individual 
    #   try/excepts so all those that can be reported, will be reported, without causing an 
    #   error to occur and crash the program
    try:
        myprint("\tConfusion Matrix:\n{}".format(confusion_matrix(yTest, ret)))
    except:
        pass
    try:
        myprint("\tClassification Report:\n{}".format(classification_report(yTest, ret)))
    except:
        pass
    try:
        myprint("\tAccuracy Score: {}".format(accuracy_score(yTest, ret)))
    except:
        pass 
    try:
        myprint("\tPrecision Score: {}".format(precision_score(yTest, ret, average="binary", pos_label="bad")))
    except:
        pass 
    try:    
        myprint("\tRecall Score: {}".format(recall_score(yTest, ret, average="binary", pos_label="bad")))
    except:
        pass
    try:
        myprint("\tF1 Score: {}".format(f1_score(yTest, ret, average="binary", pos_label="bad")))
    except:
        pass
    try:
        myprint("\tExplained Variance Score: {}".format(explained_variance_score(yTest, ret)))
    except:
        pass
    try:
        myprint("\tMax Error: {}".format(max_error(yTest, ret)))
    except:
        pass
    try:
        myprint("\tR2 Score: {}".format(r2_score(yTest, ret)))
    except:
        pass 
    try:
        myprint("\tMean Gamma Deviance: {}".format(mean_gamma_deviance(yTest, ret)))
    except:
        pass

    predictMe(modelType, model, vectorizer)

#Neural Network ML Algorithm
def NN(xTrain, yTrain, xTest, yTest, vectorizer):
    myprint("Performing Neural Network\n")
    mlp = MLPClassifier(hidden_layer_sizes=(10,10,10))
    fit = mlp.fit(xTrain, yTrain)
    reviewResults("Neural Network", mlp, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#SVC Grid Search ML Algorithm
def SVCGridSearch(xTrain, yTrain, xTest, yTest, vectorizer, mt):
    myprint("Performing SVC Grid Search\n")
    svcGrid = {"C": [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128],
        "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
    if mt:
        grid = GridSearchCV(SVC(), svcGrid, refit=True, n_jobs=-1)
    else:
        grid = GridSearchCV(SVC(), svcGrid, refit=True)
    fit = grid.fit(xTrain, yTrain)
    reviewResults("SVC Grid Search", grid, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Supprt Vecter Classifier ML Algorithm
def SVMC(xTrain, yTrain, xTest, yTest, vectorizer):
    myprint("Performing SVC\n")
    svmc = SVC()
    fit = svmc.fit(xTrain, yTrain)
    reviewResults("SVC", svmc, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Support Vector Regression ML algorithm
def SVMR(xTrain, yTrain, xTest, yTest, vectorizer):
    myprint("Performing SVR\n")
    svmr = SVR()
    fit = svmr.fit(xTrain, yTrain)
    reviewResults("SVR", svmr, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#K-Nearest Neighbor
def KNearN(xTrain, yTrain, xTest, yTest, vectorizer, mt):
    myprint("Performing K-Nearest Neighbor\n")
    if mt:
        knn = KNeighborsClassifier(n_jobs=-1)
    else:
        knn = KNeighborsClassifier()
    fit = knn.fit(xTrain, yTrain)
    reviewResults("K-Nearest Neighbor", knn, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Linear Regression
def LinR(xTrain, yTrain, xTest, yTest, vectorizer, mt):
    myprint("Performing Linear Regression\n")
    if mt:
        linr = LinearRegression(n_jobs=-1)
    else:
        linr = LinearRegression()
    fit = linr.fit(xTrain, yTrain)
    reviewResults("Linear Regression", linr, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Random Forest Classifier ML algorithm
def RanF(xTrain, yTrain, xTest, yTest, vectorizer, mt):
    myprint("Performing Random Forest Classifier\n")
    if mt:
        rfc = RandomForestClassifier(n_jobs=-1)
    else:
        rfc = RandomForestClassifier()
    fit = rfc.fit(xTrain, yTrain)
    reviewResults("Random Forest Classifier", rfc, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Decision Tree Classifier
def DecT(xTrain, yTrain, xTest, yTest, vectorizer):
    myprint("Performing Decision Tree Classifier\n")
    dfc = DecisionTreeClassifier()
    fit = dfc.fit(xTrain, yTrain)
    reviewResults("Decision Tree Classifier", dfc, fit, xTrain, yTrain, xTest, yTest, vectorizer)

#Logistic Regression
def LogR(xTrain, yTrain, xTest, yTest, vectorizer, mt):
    myprint("Performing Logistic Regression\n")
    if mt:
        lgs = LogisticRegression(n_jobs=-1)
    else:
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
    myprint("Creating testing and training data\n")
    passwordsCSV = pd.read_csv(path, ',', on_bad_lines="skip", engine="python")

    yLabels = passwordsCSV["Label"]
    allPass = passwordsCSV["Password"].values.astype('U')

    vectorizer = TfidfVectorizer(tokenizer=getTokens, lowercase=False)
    x = vectorizer.fit_transform(allPass)

    xTrain, xTest, yTrain, yTest = train_test_split(x, yLabels, test_size=0.3, random_state=42)

    return xTrain, yTrain, xTest, yTest, vectorizer

#Converts good and bad to 1 and 0 for Regression algorithms
def convertToInts(yTrain, yTest):
    myprint("Converting 'good' values to 1 and 'bad' values to 0. This means that the closer the value is to 1, the better the password.")
    myprint("This step is needed for Linear Regression and Support Vector Regression\n")
    intYTrain = []
    intYTest = []
    for x in yTrain:
        if x == "good":
            intYTrain.append(1)
        else:
            intYTrain.append(0)
    for x in yTest:
        if x == "good":
            intYTest.append(1)
        else:
            intYTest.append(0)

    return intYTrain, intYTest

def forceOptions(value):
    val = int(value)
    if val >= 0 and val < 10:
        return val 
    raise argparse.ArgumentTypeError("\nERROR: {} is an invalid option".format(val))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Password ML", 
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=textwrap.dedent("""Use passworddataset.csv with machine learning algorithms to determine whether passwords are good or bad
Machine Learning algorithm available for use in ml flag
\t1 - Logistic Regression
\t2 - Random Forest Classifier
\t3 - Decision Tree Classifier
\t4 - Linear Regression
\t5 - K-Nearest Neighbors
\t6 - Support Vector Regression
\t7 - Support Vector Classifier
\t8 - SVC Grid Search
\t9 - Neural Network
"""))
    parser.add_argument("-m", "--ml", help="Machine Learning algorithm to run dataset through", required=True, type=forceOptions)
    parser.add_argument("-f", "--file", help="Dataset to use", required=True)
    parser.add_argument("-o", "--outfile", help="Save output to file")
    parser.add_argument("-t", "--multithread", help="Indicator to multithread algorithms. If set to True, algorithms use all available processes", default="false")
    args = parser.parse_args()

    path = args.file

    if args.outfile is not None:
        outfile = args.outfile

    myprint("""*******************************************
Running Password Machine Learning Program
*******************************************\n\n""")

    mt = False 

    if args.multithread.lower() == "true":
        myprint("Multithreading enabled\n")
        mt = True

    if not os.path.exists(path):
        myprint("Couldn't find {}".format(args.file))
        sys.exit()

    xTrain, yTrain, xTest, yTest, vectorizer = createTrainTestData(path)

    if args.ml == 0 or args.ml == 4 or args.ml == 6:
        intYTrain, intYTest = convertToInts(yTrain, yTest)

    if args.ml == 1 or args.ml == 0:
        #Logistic Regression
        start = time.time()
        LogR(xTrain, yTrain, xTest, yTest, vectorizer, mt)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 2 or args.ml == 0:
        #Random Forest Classifier
        start = time.time()
        RanF(xTrain, yTrain, xTest, yTest, vectorizer, mt)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 3 or args.ml == 0:
        #Decision Tree Classifier
        start = time.time()
        DecT(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 4 or args.ml == 0:
        #Linear Regression
        start = time.time()
        LinR(xTrain, intYTrain, xTest, intYTest, vectorizer, mt)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 5 or args.ml == 0:
        #K-Nearest Neighbors
        start = time.time()
        KNearN(xTrain, yTrain, xTest, yTest, vectorizer, mt)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 6 or args.ml == 0:
        #Suport Vector Regression
        start = time.time()
        SVMR(xTrain, intYTrain, xTest, intYTest, vectorizer)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 7 or args.ml == 0:
        #Support Vector Classifier
        start = time.time()
        SVMC(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 8 or args.ml == 0:
        #SVC Grid Search
        start = time.time()
        SVCGridSearch(xTrain, yTrain, xTest, yTest, vectorizer, mt)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
    if args.ml == 9 or args.ml == 0:
        #Neural Network
        start = time.time()
        NN(xTrain, yTrain, xTest, yTest, vectorizer)
        stop = time.time()
        myprint("\tTime took to complete: {:.2f} seconds\n".format(stop-start))
        myprint("****************************************\n")
