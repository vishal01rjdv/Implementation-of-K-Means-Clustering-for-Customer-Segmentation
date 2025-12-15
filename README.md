# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1. Import required libraries
2. Load the dataset (Mall_Customers.csv)
3. Check dataset info,Check for missing values
4. Select features: Annual Income and Spending Score
5. Standardize the selected features
6. Apply Elbow method for k = 1 to 10 and Plot
7. Apply Silhouette Score for k = 2 to 10 and Plot 
8. Fit K-Means model
9. Predict cluster labels and add cluster labels to the dataset
10.Compute cluster centers
11. Visualize clusters using scatter plot

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:VISHAL R 
RegisterNumber:25004464

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------
# 1. Load the dataset
# ---------------------------------------
df = pd.read_csv("Mall_Customers.csv")  # UPDATE PATH IF NEEDED
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
display(df.head())

# ---------------------------------------
# 2. Check info and missing values
# ---------------------------------------
print("\nDataset Info:")
display(df.info())
print("\nMissing Values:")
display(df.isnull().sum())

# ---------------------------------------
# 3. Select features for clustering
# Using Income & Spending Score
# ---------------------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

print("\nFeatures Used:", features)

# ---------------------------------------
# 4. Standardize the data
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 5. Elbow Method to choose k
# ---------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 6. Silhouette Scores
# ---------------------------------------
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), sil_scores, marker='o', color="orange")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 7. Apply KMeans (Choose k=5 by elbow)
# ---------------------------------------
k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# ---------------------------------------
# 8. Cluster Centers in original units
# ---------------------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers_original, columns=features)
centers_df["Cluster"] = range(k_final)

print("\nCluster Centers (Original Values):")
display(centers_df.round(2))

# ---------------------------------------
# 9. Visualization of Clusters
# ---------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=70
)

# Show cluster centers
plt.scatter(
    centers_df["Annual Income (k$)"],
    centers_df["Spending Score (1-100)"],
    s=250,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means (k=5)")
plt.legend()
plt.grid(True)
plt.show()
 
*/
```

## Output:


<img width="966" height="673" alt="Screenshot 2025-12-15 104912" src="https://github.com/user-attachments/assets/ecfacbe9-f2b5-44bb-a672-9a2d3f55d8a7" />


<img width="760" height="300" alt="Screenshot 2025-12-15 104922" src="https://github.com/user-attachments/assets/a3fe4128-085c-434f-b7a1-5b0918e64ab6" />


<img width="540" height="391" alt="image" src="https://github.com/user-attachments/assets/7d269704-3489-4255-889a-4a4c169e6981" />


<img width="545" height="391" alt="image" src="https://github.com/user-attachments/assets/300fb740-5bac-44ab-b9f4-2d25bbe46317" />


<img width="846" height="477" alt="Screenshot 2025-12-15 104930" src="https://github.com/user-attachments/assets/ee1f4551-0f3e-4f37-8fd3-598f1d40c3c2" />


<img width="695" height="545" alt="image" src="https://github.com/user-attachments/assets/c1be4081-834b-4517-8182-89a695454dd5" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
