
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/iris.csv")

os.makedirs("output", exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


# ==========================================
# 2. KMeans Function
# ==========================================

def run_kmeans(features, title, filename):

    # Select features
    X = df[features].copy()

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Clustering
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    # Convert Centroids to Original Scale
    centers = scaler.inverse_transform(
        kmeans.cluster_centers_
    )

    # ==========================================
    # 3. Create Results DataFrame
    # ==========================================

    results = X.copy()

    results["variety"] = df["variety"].values
    results["cluster"] = clusters

    # Distance to nearest centroid
    distances = kmeans.transform(X_scaled).min(axis=1)

    results["Distance to Centroid"] = distances

    # Save CSV
    results.to_csv(
        f"output/{filename}.csv",
        index=False
    )

    # ==========================================
    # 4. Cluster Statistics
    # ==========================================

    counts = results["cluster"].value_counts().sort_index()

    print("\n" + "=" * 45)
    print(title)
    print("=" * 45)

    print("\nCluster Counts:")
    print(counts)

    print("\nCentroids:")
    print(
        pd.DataFrame(
            centers,
            columns=features
        )
    )

    print("\nFirst 10 Results:")
    print(results.head(10))

    # ==========================================
    # 5. Visualization
    # ==========================================

    fig, ax = plt.subplots(figsize=(11, 7))

    palette = sns.color_palette("Set2", 3)

    # Plot Data Points
    for cluster_id in range(3):

        mask = clusters == cluster_id

        ax.scatter(
            X.loc[mask, features[0]],
            X.loc[mask, features[1]],
            color=palette[cluster_id],
            s=85,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=f"Cluster {cluster_id} (n={counts[cluster_id]})"
        )

    # Plot Centroids
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        color="red",
        marker="X",
        s=320,
        edgecolors="black",
        linewidths=1.2,
        label="Centroids",
        zorder=10
    )

    # Centroid Labels
    for i, center in enumerate(centers):

        ax.annotate(
            f"C{i}",
            xy=(center[0], center[1]),
            xytext=(9, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold"
        )

    # Titles and Labels
    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        features[0],
        fontsize=13
    )

    ax.set_ylabel(
        features[1],
        fontsize=13
    )

    ax.legend(
        title="Clusters",
        fontsize=10,
        title_fontsize=11
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save Plot
    plt.savefig(
        f"output/{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nCSV saved: output/{filename}.csv")
    print(f"Plot saved: output/{filename}.png")


# ==========================================
# 6. Sepal Clustering
# ==========================================

run_kmeans(
    features=["sepal.length", "sepal.width"],
    title="KMeans Clustering - Sepal",
    filename="sepal_kmeans"
)


# ==========================================
# 7. Petal Clustering
# ==========================================

run_kmeans(
    features=["petal.length", "petal.width"],
    title="KMeans Clustering - Petal",
    filename="petal_kmeans"
)


# ==========================================
# 8. Finished
# ==========================================

print("\nAll Clustering Tasks Completed!")
print("Check the output folder.")
