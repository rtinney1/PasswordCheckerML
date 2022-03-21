# PasswordCheckerML

## Setting up the environment
Machine Learning programs to verify whether a password is good or bad. 

First install the required modules via
```
pip install -r requirements.txt
```

then set the following environment variables

### Windows
```
set FLASK_APP=main.py
set FLASK_ENV=development
```

### Linux
```
export FLASK_APP=main.py
export FLASK_ENV=development
```

To run the program, type
```
flask run
```

You can access the web application via http://127.0.0.1:5000 in a browser.

Download the RockYou2021 dataset [here](https://download2390.mediafire.com/zuxu7c9mngdg/rjt6ytgs9b3scht/RockYou2021.txt.gz) 

Once the RockYou2021 dataset is downloaded, run the python script `splitme.py` to split the RockYou2021, realuniq.lst, into it's separate parts. To run the script, make sure the `splitme.py` text file is within the same directory as the realuniq.list file (please refer to the required folder structure below). This greatly reduces the amount of time it takes to search through the entire dataset because it separates the data by the first character of the string and the number of characters within the string.

The folder structure for this project must look like
- Directory
  - templates
    - creator.html
    - checker.html
  - RockYou2021.txt
    - rockyou2021_part1_8.txt
    - rockyou2021_part2_8.txt
    - (and so on. There should be 24,500 files in this folder)
  - main.py
  - websitedataset2.csv
  
## Webpages
### Checker
The program will not allow you to check a password unless it meets the following criteria:
  - More than 8 characters long
  - Contains uppercase characters
  - Contains lowercase characters
  - Contains numbers
  - Contains special characters

Once a password meets the criteria, the user can press the Check button to see what the machine learning algorithm predicts. It will also search through the RockYou2021 password drop to see if the password is located in the file as well.
  
### Creator
The program will allow the user to enter in a password, phrase, word, or series of characters/numbers and attempts to create a strong password from what was entered. Once the machine learning algorithm determines that the created password is strong, the program tries to find the created password within the RockYou2021 password drop.

### Generate
This program will allow the user to enter a topic of their choice in which to create a password from. The program uses the Wikipedia API to grab the contents of the most relevent page. From the page, it will then use the NLTK python module to grab all proper nouns. Four random words will be selected from the list and create a space separated password that will, hopefully, be easy for the user to remember. Once the password is created, the program will try to find the generated password within the RockYou2021 password drop.

A log of these created passwords are stored in passwordlog.log

The machine learning algorithm in use is Logistic Regression

Currently, the dataset is a hardset value on line 32 in main.py. This dataset (websitedataset2.csv) is currently found within the repo. If you want to generate a new dataset, please change to the dataset created via the createPasswordDataSet.py script

## Expirement
The main expirement was to see if the machine learning algorithm could create strong passwords from memorable passphrases. 

Once the environment is set up, run 

```
python testPassphrase.py
```
This will pass 150 randomly generated words (located in expirmentPass.txt) to the webserver, 100 times.
