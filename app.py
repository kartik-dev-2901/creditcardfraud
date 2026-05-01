import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Load model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

# -----------------------------
# 📊 DATASET DISTRIBUTION GRAPH
# -----------------------------
st.subheader("📊 Dataset Distribution")

df = pd.read_csv("creditcard.csv")
counts = df["Class"].value_counts()

fig, ax = plt.subplots()
ax.bar(["Normal", "Fraud"], counts.values)
ax.set_title("Transaction Distribution")

st.pyplot(fig)
st.subheader("📊 Confusion Matrix")

# Prepare same features used in training
df_model = df[["V1", "V2", "V3", "Amount", "Class"]]

X = df_model.drop("Class", axis=1)
y = df_model["Class"]

# Take small sample (for speed)
X_sample = X.sample(2000, random_state=42)
y_sample = y.loc[X_sample.index]

# Predictions
y_pred = model.predict(X_sample)

# Confusion matrix
cm = confusion_matrix(y_sample, y_pred)

# Plot
fig3, ax3 = plt.subplots()
ax3.imshow(cm)

ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_title("Confusion Matrix")

# Labels
ax3.set_xticks([0,1])
ax3.set_yticks([0,1])
ax3.set_xticklabels(["Normal", "Fraud"])
ax3.set_yticklabels(["Normal", "Fraud"])

# Numbers inside boxes
for i in range(2):
    for j in range(2):
        ax3.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig3)
# -----------------------------
# USER INPUT
# -----------------------------
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
amount = st.slider("Transaction Amount", 0, 5000)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([[v1, v2, v3, amount]],
                              columns=["V1", "V2", "V3", "Amount"])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: {prob:.2f}")

    # -----------------------------
    # 📈 PROBABILITY GRAPH
    # -----------------------------
    st.subheader("📈 Fraud Probability Graph")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Fraud Probability"], [prob])
    ax2.set_ylim(0, 1)

    st.pyplot(fig2)

    # -----------------------------
    # ⚠️ RISK INDICATOR
    # -----------------------------
    st.subheader("⚠️ Risk Level")

    if prob > 0.7:
        st.warning("High Risk Transaction")
        st.progress(100)
    elif prob > 0.3:
        st.info("Medium Risk")
        st.progress(50)
    else:
        st.success("Low Risk")
        st.progress(10)

# -----------------------------
# FOOTER INFO
# -----------------------------
st.write("📌 Model: Random Forest (Balanced Dataset)")
st.write("🔍 Note: Model trained using selected features for fast prediction")