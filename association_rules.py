import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(data):

    subset = data[["Amount", "Class"]].copy()

    subset["Amount"] = pd.qcut(
        subset["Amount"], 4, labels=["Low", "Medium", "High", "VeryHigh"]
    )

    encoded = pd.get_dummies(subset)

    frequent = apriori(encoded, min_support=0.01, use_colnames=True)

    rules = association_rules(
        frequent, metric="confidence", min_threshold=0.5
    )

    return rules