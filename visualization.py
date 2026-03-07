import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

def fraud_distribution(data):

    print(data["Class"].value_counts())

    sns.countplot(x="Class", data=data)

    plt.title("Fraud vs Normal Transactions")
    plt.xlabel("Class (0 = Normal, 1 = Fraud)")
    plt.ylabel("Count")

    # make the fraud bar visible
    plt.yscale("log")

    plt.show()