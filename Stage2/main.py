"""
File: main.py
Creator: Randi Tinney
Date Created: 27 Dec 2021
Creates a flask webserver for password checking/creating. 
Uses Logistic Regression to see if passwords are strong based off of good/bad dataset. 
Searches through RockYou2021 password drop to see if password exists
"""

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

import pandas as pd

import os
import random
import re
import threading


app = Flask(__name__, template_folder="templates")

"""
Class for logistic regression methods. Ensures machine learning algorithm is only trained once at the start of the program
"""
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

    #Checks to see if passed password is good or bad
    def checkPassword(self, password):
        predict = []
        predict.append(password)

        xPredict = self.vectorizer.transform(predict)
        yPredict = self.lgs.predict(xPredict)

        return yPredict[0]

"""
This function locks and releases the log file so multiple
threads do no write to it at the same time
"""
loglock = threading.Lock()

def write_to_file(file, text):
    #print("Waiting to write to log")
    loglock.acquire() # thread blocks at this line until it can obtain lock
    #print("Writing to log")
    # in this section, only one thread can be present at a time.
    for t in text:
        with open(file, "a") as f:
            f.write(t)

    loglock.release()

"""
Class for the password checking/creating thread. Helps the website update user with current progress
"""
class PasswordThread(threading.Thread):
    def __init__(self, log, lgr, logText, runner="create", password="", caps=False, backnum=False, frontnum=False, special=False, leet=False):
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
        self.logText = logText

        self.rockYouRes = ""
        self.results = ""

        super().__init__()

    def createPasswords(self, word):
        i = 0
        password = word
        while True:
            i += 1
            result = self.lgr.checkPassword(password)
            self.results.append([password, result])

            if result == "good":
                now = datetime.now()
                self.logText.append("{} - LogR found {} to be a good password\n".format(now.strftime("%d/%m/%Y %H:%M:%S.%f"), password))
                return password
            if i == 100:
                break
            password = self.englishToLeetspeak(password)
        return ""

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
            #print("Starting")
            self.results = []
            self.response = "Creating strong password"
            self.progress = 0
            good = False
            
            rockYouRes = ""

            strongPass = self.createPasswords(self.password)
            
            if strongPass != "":
                self.response = "Searching for password in RockYou2021"
                self.progress = 50
                ret = self.findInRockYou2021(strongPass)
                if ret:
                    now = datetime.now()
                    self.logText.append("{} - {} was found in the RockYou2021.txt password drop\n".format(now.strftime("%d/%m/%Y %H:%M:%S.%f"), strongPass))
                    self.rockYouRes = "{} was found in the RockYou2021.txt password drop".format(strongPass)
                else:
                    now = datetime.now()
                    self.logText.append("{} - {} was NOT found in the RockYou2021.txt password drop\n".format(now.strftime("%d/%m/%Y %H:%M:%S.%f"), strongPass))
                    self.rockYouRes = "{} was NOT found in the RockYou2021.txt password drop".format(strongPass)
            else:
                self.rockYouRes = "No good password was able to be created"

            write_to_file(self.log, self.logText)
            #print(self.rockYouRes)
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
            if re.search('[a-zA-Z]', message) is not None:
                if random.random() <= 0.50:
                    if re.search("[A-Z]", message) is not None:
                        while True:
                            randNum = random.randrange(0, len(message))
                            if message[randNum].isupper():
                                message = message[:randNum] + message[randNum].lower() + message[randNum+1:]
                                break
                else:
                    if re.search("[a-z]", message) is not None:
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

    def checkMachineLearning(self):
        results = self.lgr.checkPassword(self.password)
        return results

    def findInRockYou2021(self, word):
        #fileLocks[fileNum].acquire()
        length = len(word)
        start = word[0]
        ret = False
        try:
            with open(os.path.join("Stage2", "RockYou2021.txt", "rockyou2021_part{}_{}.txt".format(start, length)), encoding="utf-8", errors="ignore", mode="r") as f:
                for line in f:
                    if word == line.replace("\n", ""):
                        ret = True
        except:
            print("No file called rockyou2021_part{}_{}.txt was found".format(start, length))
            ret = False
        #fileLocks[fileNum].release()
        return ret

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

    try:
        caps = request.form.get("caps")
        if caps is None:
            caps = True
        backNum = request.form.get("backNum")
        if backNum is None:
            backNum = True
        frontNum = request.form.get("frontNum")
        if frontNum is None:
            frontNum = True
        special = request.form.get("special")
        if special is None:
            special = True
        leet = request.form.get("leet")
        if leet is None:
            leet = True
    except:
        caps = True
        backNum = True
        frontNum = True
        special = True
        leet = True

    if len(password) > 0:
        now = datetime.now()
        logText = []
        logText.append("{} - Testing phrase: {}\n".format(now.strftime("%d/%m/%Y %H:%M:%S.%f"), password.rstrip()))
        threadID = random.randint(1,99)
        threads[threadID] = PasswordThread(log=logMe, logText=logText, runner="create", lgr=lgr, password=password, caps=caps, backnum=backNum, frontnum=frontNum, special=special, leet=leet)
        threads[threadID].start()
    else:
        threadID = -1

    return render_template("creator.html", threadID=threadID)
