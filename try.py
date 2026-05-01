import pandas as pd

df = pd.read_csv("creditcard.csv")

# Legitimate example
print(df[df["Class"] == 0].iloc[0])

# Fraud example
print(df[df["Class"] == 1].iloc[0])