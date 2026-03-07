from sklearn.cluster import KMeans


def run_kmeans(X):

    kmeans = KMeans(n_clusters=2, random_state=42)

    clusters = kmeans.fit_predict(X)

    return clusters