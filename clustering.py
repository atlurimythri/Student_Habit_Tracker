import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_clustering(n_clusters=3):

    data = pd.read_csv("processed_student_data.csv")

    features = [
        "study_hours",
        "pre_test",
        "post_test",
        "attention_span",
        "effectiveness_score"
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(X)

    return data