import pandas as pd

# Read in the first CSV file
df1 = pd.read_csv("./data/test.csv")

# Read in the second CSV file
df2 = pd.read_csv("./data/submit.csv")

# Use the 'merge' function to combine the two DataFrames on the 'id' column
result = pd.merge(df1, df2, on='id')
print(result.head())
result.to_csv("./data/result.csv", index=False)

