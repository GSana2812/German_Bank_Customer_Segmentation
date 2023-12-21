from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.cluster import KMeans
import pandas as pd
import json
from src import scale_features

app = FastAPI()

class ClusterPrediction(BaseModel):
    Credit_amount: int
    Age:int
    Duration:int


@app.post('/predict')

# http://127.0.0.1:8000/predict
async def predictions(items: List[ClusterPrediction]):

    """
       Perform KMeans clustering on a list of data items and return cluster labels.

       Parameters:
       items (List[ClusterPrediction]): A list of data items for clustering.

       Returns:
       List[dict]: A list of dictionaries, each containing "Index" (index of data item) and "Cluster" (cluster label).
       """

    data_dict = [item.model_dump() for item in items]
    df = pd.DataFrame(data_dict)
    data_scaled = scale_features(df)
    kmeans_sel = KMeans(n_clusters = 3, random_state=19).fit_predict(data_scaled)
    labels = (kmeans_sel.labels_).tolist()

    label_dict = [{"Index":i,"Cluster":label} for i, label in enumerate(labels)]

    return label_dict







