# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: magesh.n
RegisterNumber: 212222040091
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
data.head():

![Screenshot 2023-10-15 175153](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/0132c4fa-35e0-4b03-8383-52a06a5adba9)

data.info():

![Screenshot 2023-10-15 175306](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/79c50faa-26bf-4647-9540-2dd54615a67b)

isnull() and sum():

![Screenshot 2023-10-15 175321](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/cc1a8eea-2adf-4787-b698-341ac2a2e3a1)

data value count():

![Screenshot 2023-10-15 175332](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/33705919-842d-4475-b31e-c2a75f9a09c0)

data.head() for salary:

![Screenshot 2023-10-15 175353](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/b4277edd-d240-48ca-b180-7a215c882e48)

x.head():

![Screenshot 2023-10-15 175410](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/6ccdb4f3-4ccd-45c3-a9d5-1c2486b2f030)

accuracy value:

![Screenshot 2023-10-15 175421](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/7aca4148-7acf-4d39-a9d7-50164dff6bb8)

data prediction:

![Screenshot 2023-10-15 175441](https://github.com/22008496/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119476113/0b5c5cd2-9127-4b08-aaa4-94e0e6a42197)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
