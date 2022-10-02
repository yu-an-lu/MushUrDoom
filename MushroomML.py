from tkinter import Y
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Source for the ML code: https://www.milindsoorya.com/blog/mushroom-dataset-analysis-and-classification-python

class Model:
    def __init__(self):
        self.process_data()
        self.dt = self.train()
        
    def process_data(self):
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

        # print([df_test1][0])
        # print([df_test2][0])

        #Test Case 
        self.df_test1 = labelencoder.fit_transform(df_test1)
        self.df_test2 = labelencoder.fit_transform(df_test2)

        # "class" column as numpy array.
        y = df["class"].values
        # All data except "class" column.
        x = df.drop(["class"], axis=1).values
        
        # Split data for train and test.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,random_state=42,test_size=0.2)
        
        
    def train(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.x_train, self.y_train)

        #Test Cases 
        prediction1 = dt.predict([self.df_test1])[0]
        prediction2 = dt.predict([self.df_test2])[0]


        print(prediction1) #Giving 1 (poisonous)
        print(prediction2) #Giving 0 (edible)

        print("Test Accuracy: {}%".format(round(dt.score(self.x_test, self.y_test)*100,2)))
        
        return dt
    
    def predict(self, data):
        # encode data
        labelencoder=LabelEncoder()
        data = labelencoder.fit_transform(data)
        
        # predicts
        prediction = self.dt.predict(data)
        
        return prediction

# To remove
if __name__ == "__main__":
    model = Model()