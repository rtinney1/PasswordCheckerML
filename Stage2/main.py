from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

import pandas as pd

import os
import random
import threading


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
        passwordsCSV = pd.read_csv("websitedataset2.csv", ',', on_bad_lines="skip", engine="python")

        yLabels = passwordsCSV["Label"]
        allPass = passwordsCSV["Password"].values.astype('U')

        self.vectorizer = TfidfVectorizer(tokenizer=self.getTokens, lowercase=False)
        x = self.vectorizer.fit_transform(allPass)

        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x, yLabels, test_size=0.3, random_state=42)   

    def checkPassword(self, password):
        predict = []
        predict.append(password)

        xPredict = self.vectorizer.transform(predict)
        yPredict = self.lgs.predict(xPredict)

        return yPredict[0]

class PasswordThread(threading.Thread):
    def __init__(self, log, lgr, runner="create", password="", caps=False, backnum=False, frontnum=False, special=False, leet=False):
        self.progress = 0
        self.response = ""

        self.log = log
        self.lgr = lgr
        self.runner = runner
        self.password = password
        self.caps = caps
        self.backnum = backnum
        self.frontnum = frontnum
        self.special = special
        self.leet = leet

        self.rockYouRes = ""
        self.results = ""

        super().__init__()

    def run(self):
        if self.runner == "check":
            self.response = "Checking password against machine learning algorithm"
            self.progress = 25
            self.results = "Logistic Regression says the password is: {}".format(self.lgr.checkPassword(self.password))
            
            self.response = "Searching for password in RockYou2021"
            self.progress = 50
            ret = self.findInRockYou2021(self.password)
            if ret:
                self.rockYouRes = "Password was found in the RockYou2021.txt password drop"
            else:
                self.rockYouRes = "Password was NOT found in the RockYou2021.txt password drop"
            self.response = "Done"
            self.progress = 100
        elif self.runner == "create":
            self.results = []
            i = 0
            self.response = "Creating strong password"
            self.progress = 0
            password = self.password
            good = False
            rockYouRes = ""
            while True:
                i += 1
                result = self.lgr.checkPassword(password)
                self.results.append([password, result])

                if result == "good":
                    now = datetime.now()
                    with open(self.log, "a") as f:
                        f.write("{} - LogR found {} to be a good password\n".format(now.strftime("%d/%m/%Y %H:%M:%S"), password))
                    good = True
                    break
                if i == 100:
                    break
                password = self.englishToLeetspeak(password)
            if good:
                self.response = "Searching for password in RockYou2021"
                self.progress = 50
                ret = self.findInRockYou2021(password)
                if ret:
                    now = datetime.now()
                    with open(self.log, "a") as f:
                        f.write("{} - {} was found in the RockYou2021.txt password drop\n".format(now.strftime("%d/%m/%Y %H:%M:%S"), password))
                    self.rockYouRes = "{} was found in the RockYou2021.txt password drop".format(password)
                else:
                    now = datetime.now()
                    with open(self.log, "a") as f:
                        f.write("{} - {} was NOT found in the RockYou2021.txt password drop\n".format(now.strftime("%d/%m/%Y %H:%M:%S"), password))
                    self.rockYouRes = "{} was NOT found in the RockYou2021.txt password drop".format(password)
            else:
                self.rockYouRes = "No good password was able to be created"
            self.response = "Done"
            self.progress = 100

    #Idea from https://inventwithpython.com/bigbookpython/project40.html
    def englishToLeetspeak(self, message):
        # Make sure all the keys in `charMapping` are lowercase.
        charMapping = {
            'a': ['4', '@', '/-\\'], 'c': ['('], 'd': ['|)'], 'e': ['3'],
            'f': ['ph'], 'h': [']-[', '|-|'], 'i': ['1', '!', '|'], 'k': [']<'],
            'o': ['0'], 's': ['$', '5'], 't': ['7', '+'], 'u': ['|_|'],
            'v': ['\\/']}

        if self.leet:
            for x in range(0, len(message)):  # Check each character:
                char = message[x]
                if char.lower() in charMapping:
                    possibleLeetReplacements = charMapping[char.lower()]
                    leetReplacement = random.choice(possibleLeetReplacements)
                    leetspeak = message[:x] + leetReplacement + message[x+1:]
                    return leetspeak
        
        ran = random.random()
        if ran <= 0.25 and self.frontnum:
            randNum = random.randrange(0, 10)
            message = "{}{}".format(randNum, message)
        elif ran > 0.25 and ran <= 0.50 and self.caps:
            if random.random() <= 0.50:
                while True:
                    randNum = random.randrange(0, len(message))
                    if message[randNum].isupper():
                        message = message[:randNum] + message[randNum].lower() + message[randNum+1:]
                        break
            else:
                while True:
                    randNum = random.randrange(0, len(message))
                    if message[randNum].islower():
                        message = message[:randNum] + message[randNum].upper() + message[randNum+1:]
                        break
        elif ran > 0.50 and ran <= 0.75 and self.backnum:
            randNum = random.randrange(0, 10)
            message = "{}{}".format(message, randNum)
        elif self.special:
            inserts = list("!@#$%^&*()x_-=+")
            if random.random() <= 0.50:
                message = random.choice(inserts) + message
            else:
                message += random.choice(inserts)

        return message

    def findInRockYou2021(self, word):
        print("Looking for bad passwords")
        with open(os.path.join("RockYou2021.txt", "realuniq.lst"), encoding="utf-8", errors="ignore") as f:
            for line in f:
                if word == line.replace("\n", ""):
                    return True

        return False

    def checkMachineLearning(self):
        results = self.lgr.checkPassword(self.password)
        return results

lgr = LogR()
logMe = "passwordlog.log"

threads = dict()

@app.route('/')
def home():
    return render_template("checker.html", threadID=-1)

@app.route("/checkPass/", methods=["POST"])
def checkPass():
    global lgr
    global threads

    password = request.form["password"]
    threadID = random.randint(1,99)
    threads[threadID] = PasswordThread(runner="check", lgr=lgr, password=password)
    threads[threadID].start()

    return render_template("checker.html", threadID=threadID)

@app.route("/creator/")
def creator():
    return render_template("creator.html", results=dict())

@app.route("/checker/")
def checker():
    return render_template("checker.html", threadID="-1", results="")

@app.route("/progress/", methods=["POST"])
def getProgress():
    global threads
    threadID = request.form["threadID"]
    thread = threads[int(threadID)]

    progress = thread.progress
    response = thread.response
    results = thread.results
    rockYou = thread.rockYouRes

    if progress >= 100:
        del threads[int(threadID)]
        threadID = -1
    return {"progress": progress, "response": response, "results": results, "rockYouRes": rockYou, "threadID": threadID}

@app.route("/createPass/", methods=["POST"])
def createPass():
    global lgr
    global threads
    global logMe

    password = request.form["password"]
    caps = request.form.get("caps")
    backNum = request.form.get("backNum")
    frontNum = request.form.get("frontNum")
    special = request.form.get("special")
    leet = request.form.get("leet")

    threadID = random.randint(1,99)
    threads[threadID] = PasswordThread(log=logMe, runner="create", lgr=lgr, password=password, caps=caps, backnum=backNum, frontnum=frontNum, special=special, leet=leet)
    threads[threadID].start()

    return render_template("creator.html", threadID=threadID)
