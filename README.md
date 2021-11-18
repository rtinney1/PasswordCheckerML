# PasswordCheckerML
Machine Learning programs to verify whether a password is good or bad. Each script (Python and R) use a variety of machine learning algorithms to determine which provides the highest accuracy, precision, and other metrics.

The dataset is created with the createPasswordDataSet.py script. It uses the rockyou.txt file to formulate a list of "bad" passwords and uses an internal function to create "good" passwords based off of the following criteria:
1. Must be greater than 8 characters long (we force the good passwords to be a length between 8 and 12)
2. Contain at least 3 of the 4:
   - Lowercase Character
   - Uppercase Character
   - Number
   - Special Character

Once the data set has been created, users then have the option to create an already tokenized file (required for R script). This gives each type of character a numerical value for the machine learning algorithms to work with. The Python script does this automatically.

Currently, the following algorithms are available in the

###### Python script
1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier
4. Linear Regression
5. K-Nearest Neighbors
6. Support Vector Regression
7. Support Vector Classifier
8. SVC Grid Search
9. Neural Net

###### R script
1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Model
4. K-Nearest Neighbors
5. SVM (aka Support Vector Classifier)
6. SVM Grid Search
7. Neural Net

## createPasswordDataSet.py
Download rockyou.txt via [this link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf2ceg4vDzAhUEZzABHcQTAI4QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&usg=AOvVaw3snAERl1mU6Ccr4WFEazBd)

Make sure rockyou.txt is in the same directory as createPasswordDataSet.py. 

To run, type
```
python createPasswordDataSet.py -n NUM -f FILE -t TOKENIZE(True/False)
```
where 
 - NUM is the number of password you want in the created dataset
 - FILE is the name of the file without the .csv portion
 - TOKENIZE is a True/False value on whether you want the tokenized csv files to be created. Default is False. This will create two extra files along with the FILE.csv. It will create a FILE_tfidf_char.csv and FILE_tfidf_int.csv.

**Note**: The number passed will actually result in that number * 2 passwords. The number indicates the number of "bad" passwords to grab from rockyou.txt and the script will create an equal number of "good" passwords. For example, if you pass 10 to the script, you will get 20 passwords.

**Note**: If you pass the word *all* to the script, it will use all viable passwords from rockyou.txt. This creates a dataset of around 2.5 million "good" and "bad" passwords.

**Note**: For the R script to work, you need the tokenized files and need to pass True to the -t flag
  
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
8. SVC Grid Search
9. Neural Net
  
## passwordML.R
Need to change the hardcoded location (line 11 for the tfidf_char) of the tokenized file created from the createPasswordDataSet.py. This enables the various machine learning algorithms to function correctly and give accurate metrics. 

Script reports times (if applicable), confusion matrix, calculated precision, accuracy, recall, and F1 scores, and any plots that were deemed useful. Depending on the size of the dataset used, algorithms can take hours and/or days.

