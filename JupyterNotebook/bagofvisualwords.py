import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA
import imagedataset

def Dictionary(descriptors, N):
    kmeans = KMeans(n_clusters=N, max_iter=2000).fit(descriptors)
    print(kmeans)
    centers = kmeans.cluster_centers_
    return centers

def LoadKmeans(folder = ""):
    print("Loading kmeans...")
    kmeans = np.load("bovw_centers.kmeans.npy")
    print("bovw_centers.kmeans Loaded !")
    return kmeans

def GenerateKmeans(words, N):
    print("Training GMM of size", N)
    kmeans = Dictionary(words, N)
    print("Saving kmeans at: bovw_centers.kmeans.npy")
    np.save("bovw_centers.kmeans", kmeans)
    return kmeans

def BovwFeatures(annotations, kmeans, sc_descriptors):
    vector_len = kmeans.shape[0]
    kmeans = (kmeans,)
    X,y = imagedataset.GenerateFeatures(BovwVector, annotations, vector_len, kmeans, sc_descriptors)
    print("Features generated")
    return X, y
    
def BovwVector(samples, centers):
    predictions = []
    for descriptor in samples:
        prediction = np.argmin([LA.norm(distance) for distance in centers-descriptor])
        predictions.append(prediction)
    hist, _ = np.histogram(predictions, bins=np.arange(centers.shape[0]+1))
    return hist