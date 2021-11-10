"""
File: createPasswordDataSet.py
Creator: Randi Tinney
Date Created: 29 Oct 2021
Uses the rockyou.txt file to create a csv of 'bad' passwords. Then uses a password creating function to create 'good' passwords.
Creates the passworddataset.csv file to be used for ML
"""

import argparse, csv, random, string, sys

charsUpper = list(string.ascii_uppercase)
charsLower = list(string.ascii_lowercase)
digits = list(string.digits)
specialChars = list("!@#$%^&*()")

contain = ["charsUpper", "charsLower", "digits", "specialChars"]

"""Good passwords are greater than 8 characters and contain at least 3 of the following:
    lowercase
    uppercase
    digit
    special character
getGoodPassword generates a password with length of 8-12 characters
and will choose the characters randomly from the chars global list.
"""
def getGoodPassword():
    passLengthTotal = random.randint(8,13)
    #print("Total password length: {}".format(passLengthTotal))
    password = []

    #shuffle contain list so requirements are random
    random.shuffle(contain)

    #make sure to get at least one of the requirement, but not fill password
    length = random.randint(1, passLengthTotal-2)
    #print("Length of first requirement: {}".format(length))

    if contain[0] == "charsUpper":
        for i in range(length):
            password.append(random.choice(charsUpper))
    elif contain[0] == "charsLower":
        for i in range(length):
            password.append(random.choice(charsLower))
    elif contain[0] == "digits":
        for i in range(length):
            password.append(random.choice(digits))
    elif contain[0] == "specialChars":
        for i in range(length):
            password.append(random.choice(specialChars))

    #temp variable for new needed length
    temp = passLengthTotal - length

    #get next number of another requirement, but not fill password
    length = random.randint(1, temp-1)
    #print("Length of second requirement: {}".format(length))

    if contain[1] == "charsUpper":
        for i in range(length):
            password.append(random.choice(charsUpper))
    elif contain[1] == "charsLower":
        for i in range(length):
            password.append(random.choice(charsLower))
    elif contain[1] == "digits":
        for i in range(length):
            password.append(random.choice(digits))
    elif contain[1] == "specialChars":
        for i in range(length):
            password.append(random.choice(specialChars))

    #temp variable for new needed length
    temp = temp - length

    #fill password with the rest of the last requirement
    length = temp
    #print("Length of third requirement: {}".format(length))

    if contain[2] == "charsUpper":
        for i in range(length):
            password.append(random.choice(charsUpper))
    elif contain[2] == "charsLower":
        for i in range(length):
            password.append(random.choice(charsLower))
    elif contain[2] == "digits":
        for i in range(length):
            password.append(random.choice(digits))
    elif contain[2] == "specialChars":
        for i in range(length):
            password.append(random.choice(specialChars))

    random.shuffle(password)

    #print("Length of created password: {}".format(len(password)))

    return "".join(password)

"""
getBadPasswords reads in an array of numbers and iterates through the password file rockyou.txt 
    and appends the found password to an array. The function then returns the array
    To narrow down on passwords, passwords must be greater than 2 chars, but less than 12, they
    cannot contain a " or a space as a character.
"""
def getBadPasswords(nums):
    #read in rockyou.txt and label all passwords within as 'bad'
    try:
        badRows = []
        with open("rockyou.txt", "r", encoding="utf-8", errors="ignore") as badPassFile:
            count = 0
            for line in badPassFile:
                try:
                    if line.strip() != "" and len(line.strip()) > 2 and len(line.strip()) < 12 and "\"" not in line.strip() and " " not in line:
                        if count in nums:
                            badRows.append(line.strip())
                            count += 1
                        else:
                            count += 1
                except:
                    pass
        return badRows
    except FileNotFoundError:
        print("Please go download rockyou.txt and move to the current directory.\n\nIf within a Kali image, run the commands `cp /usr/share/wordlists/rockyou.txt.gz .' and 'gunzip rockyou.txt'\n\nOr, you can download the file from this link: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf2ceg4vDzAhUEZzABHcQTAI4QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&usg=AOvVaw3snAERl1mU6Ccr4WFEazBd")
        sys.exit()

"""
    Main function of the program
"""
def main(num, file):
    random.seed(69)
    #create dataset csv
    datasetFile = open("{}.csv".format(file), "w", newline="", encoding="utf-8")
    writer = csv.writer(datasetFile, delimiter=",")
    writer.writerow(["Password", "Label"])

    if num == "all":
        count = 0
        #read in rockyou.txt and label all passwords within as 'bad'
        try:
            badPassFile = open("rockyou.txt", "r", encoding="utf-8", errors="ignore") 
        except FileNotFoundError:
            print("Please go download rockyou.txt and move to the current directory.\n\nIf within a Kali image, run the commands `cp /usr/share/wordlists/rockyou.txt.gz .' and 'gunzip rockyou.txt'\n\nOr, you can download the file from this link: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf2ceg4vDzAhUEZzABHcQTAI4QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&usg=AOvVaw3snAERl1mU6Ccr4WFEazBd")
            sys.exit()

        for line in badPassFile:
            try:
                if line.strip() != "" and len(line.strip()) > 2 and len(line.strip()) < 12 and "\"" not in line.strip() and " " not in line:
                    badRow = [line.strip(), "bad"]
                    writer.writerow(badRow)
                    count += 1
            except:
                pass

        badPassFile.close()
        
        #create strong passwords and add as 'good'
        for x in range(count):
            goodPass = getGoodPassword()
            goodRow = [goodPass, "good"]
            writer.writerow(goodRow)
    else:
        num = int(num)
        print("Grabbing {} bad passwords from rockyou.txt".format(num))
        #create array of random numbers
        randLines = []
        count = 0
        while True:
            x = random.randint(0, 12730586)
            if x not in randLines:
                randLines.append(x)
                count += 1
                if count == num:
                    break
    
        #Get randomized bad passwords
        badRows = getBadPasswords(randLines)

        print("Adding bad passwords to {}.csv".format(file))
        for badRow in badRows:
            data = [badRow, "bad"]
            writer.writerow(data)
    
        print("Creating {} good passwords and adding them to {}.csv".format(count, file))
        #create strong passwords and add as 'good'
        for x in range(count):
            goodPass = getGoodPassword()
            data = [goodPass, "good"]
            writer.writerow(data)

    datasetFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create password dataset via given number of passwords from rockyou.txt and auto generated secure passwords")
    parser.add_argument("-n", "--num", help="Number of passwords to include in file", required=True)
    parser.add_argument("-f", "--file", help="Name of dataset file to create", default="passworddataset")
    args = parser.parse_args()
    
    main(args.num, args.file)
