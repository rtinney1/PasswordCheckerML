# PasswordCheckerML
Machine Learning programs to verify whether a password is good or bad.

The dataset is created with the createPasswordDataSet.py script. It uses the rockyou.txt file to formulate a list of "bad" passwords and uses an internal function to create "good" passwords based off of the following criteria:
1. Must be greater than 8 characters long (we force the good passwords to be a length between 8 and 12)
2. Contain at least 3 of the 4:
   - Lowercase Character
   - Uppercase Character
   - Number
   - Special Character

## createPasswordDataSet.py
Download rockyou.txt via [this link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf2ceg4vDzAhUEZzABHcQTAI4QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fbrannondorsey%2Fnaive-hashcat%2Freleases%2Fdownload%2Fdata%2Frockyou.txt&usg=AOvVaw3snAERl1mU6Ccr4WFEazBd)

Make sure rockyou.txt is in the same directory as createPasswordDataSet.py. 

To run, type
```
python createPasswordDataSet.py -n NUM -f FILE -t TOKENIZE(True/False)
```
where 
 - NUM is the number of password you want in the created dataset
 - FILE is the name of the file to create without the .csv portion
 - TOKENIZE is a True/False value on whether you want the tokenized csv files to be created. Default is False. This will create an extra file (FILE_tfidf_char.csv) along with the FILE.csv.

**Note**: The number passed will actually result in that number * 2 passwords. The number indicates the number of "bad" passwords to grab from rockyou.txt and the script will create an equal number of "good" passwords. For example, if you pass 10 to the script, you will get 20 passwords.

**Note**: If you pass the word *all* to the script, it will use all viable passwords from rockyou.txt and an equal number of good passwords

**Note**: For the R script to work, you need the tokenized file and need to pass True to the -t flag
  
## Stage1
The following questions are explored:

1. Can machine learning algorithms predict whether passwords are good or bad
2. Which machine learning algorithm is the best in terms of time to complete, accuracy, prediction, recall, and F1
3. Which coding implementation, R or Python, produces the best results
  
## Stage2
The following questions are explored:

1. Can machine learning algorithms be applied to actual applications
2. Is a static ruleset a better judge of password strength than a machine learning algorithm
