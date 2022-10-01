import pandas as pd 

csv_directory = r"mushrooms.csv"
dataset = pd.read_csv(csv_directory)

for col in dataset: 
    print(dataset[col].unique())
