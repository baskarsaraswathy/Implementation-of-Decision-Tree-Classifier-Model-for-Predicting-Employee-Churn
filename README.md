# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: BASKAR J
RegisterNumber: 212223040025
*/

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=data['left']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
## Dataset
![image](https://github.com/user-attachments/assets/f6da97d6-4151-402d-8324-bcc1ce722580)

## Information
![image](https://github.com/user-attachments/assets/1b6e89cc-009a-4d49-b38c-820bce7baff1)

## Non Null values
![image](https://github.com/user-attachments/assets/3dfee105-73bf-47fe-898a-eb64976c6ff7)

## Encoded value
![image](https://github.com/user-attachments/assets/3aa82e6a-abde-4157-ab2a-9e89f9a4ce64)

## Count
![image](https://github.com/user-attachments/assets/176b21fb-b514-4b67-b9be-55e0bd8d1ee5)

## X and Y value
![image](https://github.com/user-attachments/assets/03269bce-290c-4696-a8c2-e92cabe029db)
![image](https://github.com/user-attachments/assets/19bfbc68-733a-46fb-8150-be29ba73fde1)


## Accuracy
![image](https://github.com/user-attachments/assets/a57f2a86-679f-49d1-a4cd-2430f35a1858)

## Predicted
![image](https://github.com/user-attachments/assets/5580e60e-a272-4c69-a4d6-927be374cf3a)

## Plot
![image](https://github.com/user-attachments/assets/adef65e0-d0c6-4450-8fc8-d6e0eb8a5a42)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
