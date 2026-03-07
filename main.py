from preprocessing import load_dataset, preprocess_data
from visualization import fraud_distribution
from classification import train_naive_bayes, train_knn, train_decision_tree
from clustering import run_kmeans
from association_rules import run_apriori
from evaluation import evaluate_model


print("Credit Card Fraud Detection Project\n")

# load data
data = load_dataset("creditcard.csv")

# visualization
fraud_distribution(data)

# preprocessing
X_train, X_test, y_train, y_test = preprocess_data(data)

# classification models
nb_model = train_naive_bayes(X_train, y_train)
knn_model = train_knn(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)

print("\nNaive Bayes Results")
evaluate_model(nb_model, X_test, y_test)

print("\nKNN Results")
evaluate_model(knn_model, X_test, y_test)

print("\nDecision Tree Results")
evaluate_model(dt_model, X_test, y_test)

# clustering
clusters = run_kmeans(X_train)
print("\nK-Means clustering completed")

# association rules
rules = run_apriori(data)

print("\nAssociation Rules")
print(rules[["antecedents", "consequents", "support", "confidence"]].head())

print("\nProject Completed")