# PasswordCheckerML
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
The folder structure for this project must look like
<Directory>
  |
  - templates
    |
    - creator.html
    |
    - checker.html
  |
  - RockYou2021.txt
    |
    - realuniq.lst
  |
  - main.py
  
## Checker
The program will not allow you to check a password unless it meets the following criteria:
  - More than 8 characters long
  - Contains uppercase characters
  - Contains lowercase characters
  - Contains numbers
  - Contains special characters

Once a password meets the criteria, the user can press the Check button to see what the machine learning algorithm predicts. It will also search through the RockYou2021 password drop to see if the password is located in the file as well.
  
## Creator
The program will allow the user to enter in a password, phrase, word, or series of characters/numbers and attempts to create a strong password from what was entered. Once the machine learning algorithm determines that the created password is strong, the program tries to find the created password within the RockYou2021 password drop. 
  
A log of these created passwords are stored in passwordlog.log

The machine learning algorithm in use is Logistic Regression

Currently, the dataset is a hardset value on line 32 in main.py. Please change to the dataset created via the createPasswordDataSet.py script
