import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/oil-synthetic-data/oil_synthetic_data.csv")

features = data[['density', 'viscosity_40C', 'viscosity_100C', 'viscosity_index',
                 'flash_point', 'pour_point', 'sulfur_content', 'api_gravity',
                 'thermal_conductivity', 'lubricity', 'carbon_residue', 'oxidation_stability']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features) #numpy array

#apply pca for dimentionality reduction
pca = PCA (n_components = 2)
pca_features = pca.fit_transform(scaled_features)
print(f"Explained varaince ratio:{pca.explained_variance_ratio_}" ) #captures how much variance each feature caught

#add noise to simulate real datasets
rng = np.random.default_rng(32)
noise = rng.normal(0.0, 0.02, pca_features.shape)
pca_features += noise

kmeans = KMeans(n_clusters = 2, init = "k-means++", random_state = 40) #initialise 4 clusters, and set them apart at a random distance with seed 40
print(kmeans)
clusters = kmeans.fit_predict(pca_features)

data['Cluster'] = clusters #append the result to the dataset

centroid = kmeans.cluster_centers_
distance = np.sqrt((pca_features - centroid[clusters]) ** 2).sum(axis = 1) ** 0.5
stray_points = distance > np.mean(distance) + 1.0 * np.std(distance) #returns a boolean array
print(stray_points)
print(data[['Sample_ID', 'Cluster']].head())
print(f"stray points : {np.sum(stray_points)}")

#Visualisation
plt.scatter(pca_features[:, 0 ], pca_features[:, 1], c= clusters,cmap = 'viridis', alpha = 0.5)
plt.scatter(pca_features[stray_points,0], pca_features[stray_points, 1], c = 'red', marker = 'x', label = 'Stray Points')
plt.xlabel('PCA Clusters')
plt.ylabel('Stray Points')
plt.title('K-Means clustering on a sythetic dataset')
plt.legend()
plt.show()
