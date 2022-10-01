import pandas as pd 

csv_directory = r"C:\Users\nikol\OneDrive\Documents\ConsumeUrDoom\mushrooms.csv"
dataset = pd.read_csv(csv_directory)

for col in dataset: 
    print(dataset[col].unique())
