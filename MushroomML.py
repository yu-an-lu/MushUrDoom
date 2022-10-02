import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Source for the ML code: https://www.milindsoorya.com/blog/mushroom-dataset-analysis-and-classification-python

df = pd.read_csv('mushrooms.csv')

df = df.astype('category')
df = df.drop(["veil-type"],axis=1)

#Getting a test case 1
df_test1 = df.iloc[0]
df_test1 = df_test1.to_numpy()
df_test1 = np.delete(df_test1, 0)
#Getting a test case 2
df_test2 = df.iloc[1]
df_test2 = df_test2.to_numpy()
df_test2 = np.delete(df_test2, 0)

labelencoder=LabelEncoder()

for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])

#Test Case 
df_test1 = labelencoder.fit_transform(df_test1)
df_test2 = labelencoder.fit_transform(df_test2)

# "class" column as numpy array.
y = df["class"].values
# All data except "class" column.
x = df.drop(["class"], axis=1).values
# Split data for train and test.
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

#Test Cases 
prediction1 = dt.predict([df_test1])[0]
prediction2 = dt.predict([df_test2])[0]

print(prediction1) #Giving 1 (poisonous)
print(prediction2) #Giving 0 (edible) 

print("Test Accuracy: {}%".format(round(dt.score(x_test,y_test)*100,2)))