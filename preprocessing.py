import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def preprocess_data(data):

    X = data.drop("Class", axis=1)
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test