#import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from google.colab import drive
drive.mount('/content/drive')

#reading the csv file
df = pd.read_csv('/content/drive/My Drive/archive/loan-train.csv')

#taking the log of the LoanAmount (the data is skewed right currently)
df['LoanAmount_log'] = np.log(df['LoanAmount'])

#replacing null values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['LoanAmount_log'].fillna(df['LoanAmount_log'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

#calculating total income based on applicant and coapplicant's combined income
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

#taking the log of the LoanAmount (the data is skewed right)
df['TotalIncome_log'] = np.log(df['TotalIncome'])

#splitting the data into independent(x) and dependent(y) variables
x = df.iloc[:, np.r_[1:5,9:11,13:15]].values
y = df.iloc[:, 12].values

#using train_test_split to split the data so that 80% is used to train the model and 20% tests accuracy
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#initializing the encoders
from sklearn.preprocessing import LabelEncoder
encodeX = LabelEncoder()
encodeY = LabelEncoder()

#encodes the qualitative variables with numbers to make it easier for the model to handle
for i in range(0,5):
  x_train[:,i] = encodeX.fit_transform(x_train[:,i])
x_train[:,7] = encodeX.fit_transform(x_train[:,7])

#encodes again
y_train = encodeY.fit_transform(y_train)

#encodes the test data
for j in range(0,5):
  x_test[:,j] = encodeX.fit_transform(x_test[:,i])
x_test[:,7] = encodeX.fit_transform(x_test[:,7])

#encodes again
y_test = encodeY.fit_transform(y_test)

#initializing StandardScaler
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

#using StandardScaler to make all the data standard deviations
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

#importing the Gaussian Naive Bayes machine learning model
from sklearn.naive_bayes import GaussianNB
nbclassifier = GaussianNB()

#training the model
nbclassifier.fit(x_train,y_train)

#generating predictions
y_pred = nbclassifier.predict(x_test)

#checking the accuract of the model
from sklearn import metrics
print("The GaussianNB algorithm's accuracy is: ", metrics.accuracy_score(y_pred, y_test))

filename = 'drive/My Drive/finalized_model.sav'
pickle.dump(nbclassifier, open(filename, 'wb'))