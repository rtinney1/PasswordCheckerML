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
To run passwordML.py, first install the required modules via
```
pip install -r requirements.txt
```
  
## passwordML.R
Currently unverified due to lack of machine resources
