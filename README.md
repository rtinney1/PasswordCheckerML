# PasswordCheckerML
Machine Learning programs to verify whether a password is good or bad

## createPasswordDataSet.py
Download rockyou.txt via [this link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf2ceg4vDzAhUEZzABHcQTAI4QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&usg=AOvVaw3snAERl1mU6Ccr4WFEazBd)

Make sure rockyou.txt is in the same directory as createPasswordDataSet.py. 

To run, type
```
python createPasswordDataSet.py -n NUM -f FILE
```
where NUM is the number of password you want in the created dataset and FILE is the name of the file without the .csv portion

**Note**: The number passed will actually result in that number * 2 passwords. The number indicates the number of "bad" passwords to grab from rockyou.txt and the script will create an equal number of "good" passwords. For example, if you pass 10 to the script, you will get 20 passwords.

**Note**: If you pass the word *all* to the script, it will use all viable passwords from rockyou.txt. This creates a dataset of around 2.5 million "good" and "bad" passwords.
  
## passwordML.py
First install the required modules via
```
pip install -r requirements.txt
```

then use the command
```
python passwordML.py -m ML -f FILE
```
where ML is the number corresponding to the Machine Learning algorithm you want to use and f is the full file name of the dataset. 

The following values are currently acceptable for the -m flag

0. All
1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier
4. Linear Regression
5. K-Nearest Neighbors
6. Support Vector Regression
7. Support Vector Classifier
  
## passwordML.R
Currently unverified due to lack of machine resources
