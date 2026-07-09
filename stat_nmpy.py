import numpy as np

data=np.array([2,3,4,3,5,6,67,5,5])


print("median ", np.median(data))
print("mean ", np.mean(data))
print("std", np.std(data))
print(" standard devitaion ", np.std(data ,ddof=1))



from scipy import stats as sp_stats

data = [2, 2, 3, 3, 5, 7]

result = sp_stats.mode(data, keepdims=True)
print("Mode:", result.mode[0], "| Count:", result.count[0])

## pandas  -----
import pandas as pd

data = pd.Series([2, 2, 3, 3, 5, 7])

print("Mean:", data.mean())
print("Median:", data.median())
print("Std Dev:", data.std())        
print("Mode:", data.mode().tolist()) # returns ALL modes as a list


##.........
df = pd.read_csv("your_data.csv")

print(df["column_name"].mean())
print(df["column_name"].median())
print(df["column_name"].mode())
print(df["column_name"].std())

# Or get everything at once:
print(df.describe())    # gives mean, std, min, max, quartiles for all numeric columns

 