import numpy as np
import pandas as pd


df = pd.read_csv(
    "combined_five_seasons_data/combined_five_seasons_data.csv", sep=",")


for (columnName, columnData) in df.iteritems():
    if (columnName == "Results"):
        print(columnData.values)

for index, row in df.iterrows():

    if df.loc[index, 'Results'] > 0:
        df.loc[index, 'Results'] = "Win"
    elif df.loc[index, 'Results'] < 0:
        df.loc[index, 'Results'] = "Loss"
    else:
        df.loc[index, 'Results'] = "Draw"
    print(df.loc[index, 'Results'])

df.to_csv("combined_five_seasons_data/combined_five_seasons_data.csv", index=False)

df = pd.read_csv(
    "combined_five_seasons_data/combined_five_seasons_data.csv", sep=",")

for (columnName, columnData) in df.iteritems():
    if (columnName == "Results"):
        print(columnData.values)
