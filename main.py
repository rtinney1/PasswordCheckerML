from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd

app = Flask(__name__, template_folder="templates")

class LogR():
    def __init__(self):
        self.createTrainTestData()
        print("Performing Logistic Regression\n")
        self.lgs = LogisticRegression()
        self.fit = self.lgs.fit(self.xTrain, self.yTrain)

        ret = self.fit.predict(self.xTest)

        print("\tConfusion Matrix:\n{}".format(confusion_matrix(self.yTest, ret)))
        print("\tClassification Report:\n{}".format(classification_report(self.yTest, ret)))

    def getTokens(self, input):
        #split all characters
        tokens = []
        for i in input:
            tokens.append(i)
        return tokens

    def createTrainTestData(self):
        print("Creating testing and training data\n")
        passwordsCSV = pd.read_csv("50000passwords.csv", ',', on_bad_lines="skip", engine="python")

        yLabels = passwordsCSV["Label"]
        allPass = passwordsCSV["Password"].values.astype('U')

        self.vectorizer = TfidfVectorizer(tokenizer=self.getTokens, lowercase=False)
        x = self.vectorizer.fit_transform(allPass)

        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x, yLabels, test_size=0.3, random_state=42)   

    def checkPassword(self, password):
        print(password)
        predict = []
        predict.append(password)

        xPredict = self.vectorizer.transform(predict)
        yPredict = self.lgs.predict(xPredict)

        print(yPredict)

        return yPredict[0]

lgr = LogR()

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/checkPass/", methods=["POST"])
def checkPass():
    password = request.form["password"]
    print(password)

    results = lgr.checkPassword(password)
    return render_template("index.html", results=results)