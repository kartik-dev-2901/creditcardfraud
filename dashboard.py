import streamlit as st
import numpy as np
import pandas as pd

from preprocessing import load_dataset, preprocess_data
from classification import train_decision_tree

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Credit Card Fraud Monitoring Dashboard")


# ------------------------------------------------
# Cache dataset loading
# ------------------------------------------------

@st.cache_data
def get_data():
    return load_dataset("creditcard.csv")


data = get_data()


# ------------------------------------------------
# Fraud Distribution Section
# ------------------------------------------------

st.header("Fraud vs Normal Transactions")

fraud_counts = data["Class"].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.metric("Normal Transactions", fraud_counts[0])

with col2:
    st.metric("Fraud Transactions", fraud_counts[1])

st.bar_chart(fraud_counts)


# ------------------------------------------------
# Cache model training
# ------------------------------------------------

@st.cache_resource
def load_model():

    data = load_dataset("creditcard.csv")

    X_train, X_test, y_train, y_test = preprocess_data(data)

    model = train_decision_tree(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy


model, accuracy = load_model()


# ------------------------------------------------
# Model Performance
# ------------------------------------------------

st.header("Model Performance")

st.write(f"Decision Tree Accuracy: {accuracy:.4f}")


# ------------------------------------------------
# Transaction Simulator
# ------------------------------------------------

def simulate_transaction():

    transaction = np.random.normal(0, 1, 30)

    # realistic time feature
    transaction[0] = np.random.randint(0, 172800)

    # realistic amount
    amount = np.random.uniform(1, 2000)
    transaction[-1] = amount

    return transaction.reshape(1, -1), transaction


# ------------------------------------------------
# Prediction Section
# ------------------------------------------------

st.header("Transaction Monitoring")

if st.button("Simulate Transaction"):

    sample, raw_transaction = simulate_transaction()

    prediction = model.predict(sample)

    probabilities = model.predict_proba(sample)

    fraud_prob = probabilities[0][1] * 100

    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Time:", int(raw_transaction[0]))

    with col2:
        st.write("Amount:", round(raw_transaction[-1], 2))

    st.subheader("Fraud Risk Score")

    st.progress(int(fraud_prob))

    st.write(f"Fraud Probability: {fraud_prob:.2f}%")

    if fraud_prob > 50:
        st.error("⚠ High Fraud Risk")
    else:
        st.success("✅ Low Fraud Risk")