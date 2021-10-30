import csv, random, string, sys

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
    passLengthTotal = random.randint(8,12)
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

def main():
    #create dataset csv
    datasetFile = open("passworddataset.csv", "w", newline="")
    writer = csv.writer(datasetFile, delimiter=",")

    count = 0
    #read in rockyou.txt and label all passwords within as 'bad'
    try:
        badPassFile = open("rockyou.txt", "r", encoding="utf-8", errors="ignore") 
    except FileNotFoundError:
        print("Please go download rockyou.txt and move to the current directory.\n\nIf within a Kali image, run the commands `cp /usr/share/wordlists/rockyou.txt.gz .' and 'gunzip rockyou.txt'\n\nOr, you can download the file from this link: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf2ceg4vDzAhUEZzABHcQTAI4QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&usg=AOvVaw3snAERl1mU6Ccr4WFEazBd")
        sys.exit()

    for line in badPassFile:
        try:
            if line.strip != "":
                badRow = [line.strip(), "bad"]
                writer.writerow(badRow)
                count += 1
        except:
            pass

    badPassFile.close()
    #create strong passwords and add as 'good'
    for x in range(count):
        goodRow = [getGoodPassword(), "good"]
        writer.writerow(goodRow)

    datasetFile.close()


if __name__ == "__main__":
    main()