import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

# ✅ Use same features as UI
df = df[["V1", "V2", "V3", "Amount", "Class"]]

# Balance dataset (important!)
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0].sample(len(fraud), random_state=42)

df_balanced = pd.concat([fraud, normal])

X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

print("Training model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

joblib.dump(model, "fraud_model.pkl")

print("✅ Model trained and saved!")