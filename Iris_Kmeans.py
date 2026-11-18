
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv("data/iris.csv")

# 2. Remove target column (No Y)
X = df.drop(columns=["variety"], errors="ignore")

# 3. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. KMeans - 3 Clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

# 5. PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PCA 1": X_pca[:, 0],
    "PCA 2": X_pca[:, 1],
    "cluster": df["cluster"].astype(str)
})

# 6. Matplotlib Visualization
plt.figure(figsize=(9, 6))

for cluster in sorted(plot_df["cluster"].unique()):
    data = plot_df[plot_df["cluster"] == cluster]

    plt.scatter(
        data["PCA 1"],
        data["PCA 2"],
        label=f"Cluster {cluster}",
        alpha=0.75
    )

plt.title("KMeans Clustering - Matplotlib")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# 7. Seaborn Visualization
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=plot_df,
    x="PCA 1",
    y="PCA 2",
    hue="cluster",
    palette="Set2",
    s=70
)

plt.title("KMeans Clustering - Seaborn")
plt.show()

# 8. Cluster Statistics
print(df["cluster"].value_counts().sort_index())

# 9. Display results
print(df.head(10))
