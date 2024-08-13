# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LAKSHMI PRIYA.V
RegisterNumber: 212223220049
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
![image](https://github.com/user-attachments/assets/78b91814-2493-482f-aa0b-e24e77a82e63)
![image](https://github.com/user-attachments/assets/fcaa1bc3-8160-42ef-bfce-8a908c39ac3f)
![image](https://github.com/user-attachments/assets/3bb23534-d3a9-4e9b-8fea-cbc739aaaab7)
![image](https://github.com/user-attachments/assets/eafa0875-262c-4078-b543-434582b5280b)
![image](https://github.com/user-attachments/assets/f050ebc9-b2b5-41ba-9198-230578a24204)
![image](https://github.com/user-attachments/assets/c6c2b26a-25e7-4128-b3bb-66972a3802da)
![image](https://github.com/user-attachments/assets/c02e6c5e-24b9-449c-87b5-db68178dc9dd)
![image](https://github.com/user-attachments/assets/05565e1b-a61c-4d4c-ad34-f9b027f2d468)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
