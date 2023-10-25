from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

def clusters_range_evaluation(clusters_range: List[int], data: pd.DataFrame)-> List[int]:

    """
        Evaluate KMeans clustering for a range of cluster numbers and return the corresponding inertias.

        Parameters:
        clusters_range (List[int]): List of the number of clusters to evaluate.
        data (pd.DataFrame): Data to perform clustering on.

        Returns:
        List[float]: List of inertias for each number of clusters in the clusters_range.
        """

    inertias = []

    for c in clusters_range:
        kmeans = KMeans(n_clusters=c, random_state=0).fit(data)
        inertias.append(kmeans.inertia_)

    return inertias

def best_silhouette_score(clusters_range: List[int], random_range: List[int], data: pd.DataFrame)-> pd.pivot_table:

    """
        Calculate the silhouette score for different combinations of cluster numbers and random seeds.

        Parameters:
        clusters_range (List[int]): List of the number of clusters to evaluate.
        random_range (List[int]): List of random seeds to use for KMeans.
        data (pd.DataFrame): Data to perform clustering on.

        Returns:
        pd.DataFrame: A pivot table with silhouette scores for each combination of cluster number and random seed.
        """

    results = []

    for c in clusters_range:
        for r in random_range:
            kmeans = KMeans(n_clusters=c, random_state=r)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            results.append([c, r, silhouette_avg])

    # Create a dataframe to store results
    result = pd.DataFrame(data = results, columns = ['n_clusters', 'seed', 'silhouette_score'])
    pivot = pd.pivot_table(result, index = 'n_clusters', columns='seed', values='silhouette_score')

    return pivot

def get_clustered_data(n_clusters: int, random_state: int, data_scaled: pd.DataFrame, initial_data: pd.DataFrame):

    """
        Cluster data using KMeans and return the data with cluster labels.

        Parameters:
        n_clusters (int): The number of clusters to create.
        random_state (int): The random seed for reproducibility.
        data_scaled (pd.DataFrame): The data to cluster.
        initial_data (pd.DataFrame): The original data.

        Returns:
        pd.DataFrame: The original data with an additional 'Cluster' column containing cluster labels.
        """

    # Perform KMeans clustering with 3 clusters and a specific random seed

    kmeans_sel = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data_scaled)
    # Get the cluster labels assigned by KMeans
    labels = pd.DataFrame(kmeans_sel.labels_)
    # Assign the cluster labels to the original cluster_data
    clustered_data = initial_data.assign(Cluster = labels)

    return clustered_data

