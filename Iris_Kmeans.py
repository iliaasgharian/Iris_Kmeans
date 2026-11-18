
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Dataset
df = pd.read_csv("data/iris.csv")

# 2. Remove Y (variety)
X = df.drop(columns=["variety"])

# 3. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. KMeans Clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

# 5. PCA (4D to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Create DataFrame for Visualization
plot_df = pd.DataFrame({
    "PCA 1": X_pca[:, 0],
    "PCA 2": X_pca[:, 1],
    "cluster": df["cluster"].astype(str)
})

# 7. Transform Cluster Centers to 2D
centers = pca.transform(kmeans.cluster_centers_)

centers_df = pd.DataFrame({
    "PCA 1": centers[:, 0],
    "PCA 2": centers[:, 1]
})

# 8. Visualization (Only One Plot)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 6))

# Data Points
sns.scatterplot(
    data=plot_df,
    x="PCA 1",
    y="PCA 2",
    hue="cluster",
    palette="Set2",
    s=70
)

# 3 Centroids
sns.scatterplot(
    data=centers_df,
    x="PCA 1",
    y="PCA 2",
    color="red",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("KMeans Clustering with 3 Centroids")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# 9. Cluster Counts
print("Cluster Counts:")
print(df["cluster"].value_counts().sort_index())

# 10. Show Results
print("\nDataset with Clusters:")
print(df.head(10))
