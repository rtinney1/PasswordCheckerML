from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter

import pandas as pd
import numpy as np

import random
import sys
import os
import math

def getTokens(input):
    #split all characters
    return list(input)

def LR(path):
    passwordsCSV = pd.read_csv(path, ',', on_bad_lines="skip", engine="python")
    passwordsData = pd.DataFrame(passwordsCSV)

    passwordsData = np.array(passwordsData)
    random.shuffle(passwordsData)

    y = [d[1] for d in passwordsData]
    corpus = [d[0] for d in passwordsData]
    print(y)
    print(corpus)
    vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = vectorizer.fit_transform(corpus)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)

    lgs = LogisticRegression()
    lgs.fit(xTrain, yTrain)
    print("Logistic Regression Prediction Score: {}".format(lgs.score(xTest, yTest)))

    return vectorizer, lgs


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "passworddataset.csv")

    print("Performing Logistic Regression")
    vectorizer, lgs = LR(path)

    predict = ["ADer1234&^!!!dsdsds", "hello", "wakawaka"]
    xPredict = vectorizer.transform(predict)
    yPredict = lgs.predict(xPredict)

    print("Prediction of {} from Logistic Regression: {}".format(predict, yPredict))