# Iris Dataset Clustering with K-Means

## Overview

This project applies **K-Means Clustering** to the Iris dataset to identify three groups of flowers based on their physical measurements.

The clustering process is performed independently using two different feature sets:

- **Sepal Clustering:** Sepal Length and Sepal Width
- **Petal Clustering:** Petal Length and Petal Width

The `variety` column is excluded from the training process to ensure that clustering is completely unsupervised.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Methodology

The project follows these steps:

1. **Load Dataset:** Read the Iris dataset using Pandas.
2. **Feature Selection:** Select Sepal and Petal measurements separately.
3. **Standardization:** Apply `StandardScaler` to normalize the features.
4. **K-Means Clustering:** Apply K-Means with 3 clusters.
5. **Centroid Calculation:** Identify the center of each cluster.
6. **Visualization:** Display data points and cluster centroids using Seaborn and Matplotlib.
7. **Export Results:** Save clustering results as separate CSV files.

## Clustering Configuration

Both models use the following configuration:

| Parameter | Value |
|---|---|
| Algorithm | K-Means |
| Number of Clusters | 3 |
| Random State | 42 |
| N_Init | 10 |
| Scaling | StandardScaler |

## Visualizations

### 1. Sepal Clustering

Uses two features:

- Sepal Length
- Sepal Width

The visualization shows how the flowers are grouped based on their sepal measurements.

### 2. Petal Clustering

Uses two features:

- Petal Length
- Petal Width

This visualization independently groups the flowers based on their petal measurements.

### Plot Details

- Different colors represent different clusters.
- Red `X` markers represent cluster centroids.
- Centroid labels identify each cluster center.
- The legend displays the number of samples in each cluster.

## Output Files

All generated files are saved in the `output/` directory.

| File | Description |
|---|---|
| `sepal_kmeans.csv` | Sepal clustering results |
| `sepal_kmeans.png` | Sepal clustering visualization |
| `petal_kmeans.csv` | Petal clustering results |
| `petal_kmeans.png` | Petal clustering visualization |

### CSV Columns

Each CSV file contains:

- Selected feature measurements
- `variety`: Original flower species (not used during training)
- `cluster`: Assigned cluster label
- `Distance to Centroid`: Distance to the nearest cluster center in standardized feature space

## Project Structure

```text
Iris-Clustering/
│
├── data/
│   └── iris.csv
│
├── output/
│   ├── sepal_kmeans.csv
│   ├── sepal_kmeans.png
│   ├── petal_kmeans.csv
│   └── petal_kmeans.png
│
├── main.py
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Run the Project

Execute the Python script:

```bash
python main.py
```

Make sure the Iris dataset is available at `data/iris.csv`.

## Important Note

K-Means is an unsupervised learning algorithm. Therefore, cluster labels (0, 1, and 2) do not necessarily correspond to the original Iris species labels.

The original `variety` column is retained only for later comparison and evaluation.

---

**Project Goal:** Compare clustering results obtained from Sepal and Petal measurements using K-Means.