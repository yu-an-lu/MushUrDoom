from tkinter import Y
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

#Source for the ML code: https://www.milindsoorya.com/blog/mushroom-dataset-analysis-and-classification-python

class Model:
    def __init__(self):
        self.process_data()
        self.dt = self.train()
        
    def process_data(self):
        df = pd.read_csv('mushrooms.csv')
        df = df.astype('category')
        df = df.drop(["veil-type"],axis=1)

        labelencoder=LabelEncoder()

        for column in df.columns:
            df[column] = labelencoder.fit_transform(df[column])

        # "class" column as numpy array.
        y = df["class"].values
        # All data except "class" column.
        x = df.drop(["class"], axis=1).values
        
        # Split data for train and test.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,random_state=42,test_size=0.2)
        
        
    def train(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.x_train, self.y_train)

        # print("Test Accuracy: {}%".format(round(dt.score(self.x_test, self.y_test)*100,2)))
        
        return dt
    
    def getStatus(self, data):
        # encode data
        labelencoder=LabelEncoder()
        data = labelencoder.fit_transform(data)
        
        # predicts
        prediction = self.dt.predict([data])[0]
        
        #poisonous = 1 
        #edible = 0 
        return prediction

# # To remove
# if __name__ == "__main__":
#     data = {'cap-shape': 'x', 'cap-surface': 's', 
#             'cap-color': 'n', 'bruises': 't', 'odor': 'p',
#             'gill-attachment': 'f', 'gill-spacing': 'c',
#             'gill-size': 'n', 'gill-color': 'k',
#             'stalk-shape': 'e', 'stalk-root': 'e',
#             'stalk-surface-above-ring': 's',
#             'stalk-surface-below-ring': 's',
#             'stalk-color-above-ring': 'w',
#             'stalk-color-below-ring': 'w',
#             'veil-color': 'w', 'ring-number': 'o',
#             'ring-type': 'p', 'spore-print-color': 'k',
#             'population': 's', 'habitat': 'u'}
    
#     data_ar = []
    
#     # get prediction
#     for key, value in data.items():
#         data_ar.append(value)

#     # use trained decision tree
#     model = Model()
#     prediction = model.getStatus(data_ar)
    
#     # return prediction
#     print(prediction)