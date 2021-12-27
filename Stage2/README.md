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

The program will not allow you to check a password unless it meets the following criteria:
  - More than 8 characters long
  - Contains uppercase characters
  - Contains lowercase characters
  - Contains numbers
  - Contains special characters

Once a password meets the criteria, the user can press the Check button to see what the machine learning algorithm predicts. 

The machine learning algorithm in use is Logistic Regression

Currently, the dataset is a hardset value on line 32. Please change to the dataset created via the createPasswordDataSet.py script
