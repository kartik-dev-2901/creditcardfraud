from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def train_naive_bayes(X_train, y_train):

    model = GaussianNB()
    model.fit(X_train, y_train)

    return model


def train_knn(X_train, y_train):

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model