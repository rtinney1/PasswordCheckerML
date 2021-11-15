# PasswordCheckerML
Machine Learning programs to verify whether a password is good or bad

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
  
## passwordML.R
Need to change the hardcoded locations (line 11 for the tfidf_char and line 12 for the tfidf_int) of the tokenized files created from the createPasswordDataSet.py. This enables the various machine learning algorithms to function correctly and give accurate metrics. 

Currently, there are two algorithms (Logistic Regression and Linear Regression) that requires the tfidf_int csv file. They give values and predictions, but currently I am unsure how to read the data. The other algorithms give good plots and confusionMatrix metrics.

The following algorithms are currently being used:
1. Logistic Regression
2. Decision Tree Model
3. Linear Regression
4. K-Nearest Neighbors
5. SVM (aka Support Vector Classifier)
6. SVM Grid Search
7. Neural Net
