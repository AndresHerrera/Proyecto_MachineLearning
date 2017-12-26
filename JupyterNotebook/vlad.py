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
    kmeans = np.load("vlad_centers.kmeans.npy")
    return kmeans

def GenerateKmeans(words, N):
    print("Training GMM of size", N)
    kmeans = Dictionary(words, N)
    print("Saving kmeans at: vlad_centers.kmeans.npy")
    np.save("vlad_centers.kmeans", kmeans)
    return kmeans

def VladFeatures(annotations, kmeans, sc_descriptors):
    vector_len = kmeans.shape[0]*128
    kmeans = (kmeans,)
    X,y = imagedataset.GenerateFeatures(VladVector, annotations, vector_len, kmeans, sc_descriptors)
    return X, y
    
def VladVector(samples, centers):
    predictions = np.concatenate(
            [np.sum((samples-center),axis=0)/(LA.norm(np.sum((samples-center),axis=0))**2) for center in centers]
            )
    return predictions