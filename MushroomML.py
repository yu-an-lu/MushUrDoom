from tkinter import Y
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random 
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
    
    def getStatus(self, data):
        # encode data
        labelencoder=LabelEncoder()
        data = labelencoder.fit_transform(data)
        
        # predicts
        prediction = self.dt.predict([data])[0]
        
        return prediction

#To remove
if __name__ == "__main__":
    # randomized attributes 
    gill_attachmentR = ['a', 'd', 'f','n']
    gill_spacingR = ['c', 'w', 'd']
    gill_sizeR = ['b', 'n']
    stalk_shapeR = ['e', 't']
    stalk_rootR = ['b', 'c', 'u', 'e', 'z', 'r']
    stalk_surfaceR = ['f','y','k','s'] #use for above and below
    stalk_colorR = ['n','b','c','g','o','p','e','w','y'] #use for above and below
    veil_colorR = ['n','o','w','y']
    ring_numberR = ['n' 'o', 't']
    ring_typeR = ['c','e','f','l','n','p','s','z']
    spore_print_colorR = ['k','n','b','h','r','o','u','w','y']
    habitatR = ['g','l','m','p','u','w','d']
    
    gill_attach = random.choice(gill_attachmentR) #Randomized 
    gill_space = random.choice(gill_spacingR) #Randomized 
    gill_size = random.choice(gill_sizeR) #Randomized 
    stalk_shape = random.choice(stalk_shapeR) #Randomized 
    stalk_root = random.choice(stalk_rootR) #Randomized 
    stalk_surface_above = random.choice(stalk_surfaceR) #Randomized 
    stalk_surface_below = random.choice(stalk_surfaceR) #Randomized 
    stalk_color_above = random.choice(stalk_colorR) #Randomized 
    stalk_color_below = random.choice(stalk_colorR) #Randomized 
    veil_color = random.choice(veil_colorR) #Randomized 
    ring_number = random.choice(ring_numberR) #Randomized 
    ring_type = random.choice(ring_typeR) #Randomized 
    spore_print_color = random.choice(spore_print_colorR) #Randomized 
    habitat = random.choice(habitatR) #Randomized 
    
    data = {'cap-shape': 'x', 'cap-surface': 's', 
            'cap-color': 'n', 'bruises': 't', 'odor': 'p',
            'gill-attachment': gill_attach, 'gill-spacing': gill_space,
            'gill-size': gill_size, 'gill-color': 'k',
            'stalk-shape': stalk_shape, 'stalk-root': stalk_root,
            'stalk-surface-above-ring': stalk_surface_above,
            'stalk-surface-below-ring': stalk_surface_below,
            'stalk-color-above-ring': stalk_color_above,
            'stalk-color-below-ring': stalk_color_below,
            'veil-color': veil_color, 'ring-number': ring_number,
            'ring-type': ring_type, 'spore-print-color': spore_print_color,
            'population': 's', 'habitat': habitat}
    
    data_ar = []
    
    # get prediction
    for key, value in data.items():
        data_ar.append(value)

    # use trained decision tree
    model = Model()
    prediction = model.getStatus(data_ar)
    
    # return prediction
    print(prediction)