import numpy as np

def simulate_transaction(amount):

    # 30 features total (same as dataset)
    transaction = np.random.normal(0, 1, 30)

    # set amount feature
    transaction[-1] = amount

    # simulate time of transaction
    transaction[0] = np.random.randint(0, 172800)

    return transaction.reshape(1, -1)