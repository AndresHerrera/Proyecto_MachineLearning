import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import imagedataset

def Dictionary(descriptors, N):
    gmm = GaussianMixture(n_components=N, verbose=2, verbose_interval=10, covariance_type='diag')
    gmm = gmm.fit(descriptors)
    print(gmm)

    return gmm.means_, gmm.covariances_, gmm.weights_

def LoadGmm(folder = ""):
    print("Loading gmm...")
    means = np.load("means.gmm.npy")
    covs = np.load("covs.gmm.npy")
    weights = np.load("weights.gmm.npy")
    return means, covs, weights

def GenerateGmm(words, N):
    print("Training GMM of size", N)
    means, covs, weights = Dictionary(words, N)
    #Throw away gaussians with weights that are too small:
    th = 0.0 / N
    means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
    covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
    weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

    print("Saving gmm at: means.gmm.npy, covs.gmm.npy, weights.gmm.npy")
    np.save("means.gmm", means)
    np.save("covs.gmm", covs)
    np.save("weights.gmm", weights)
    return means, covs, weights

def LikelihoodMoment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def LikelihoodStatistics(samples, means, covs, weights):
    gaussians, s0, s1,s2 = {}, {}, {}, {}
    
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]

    for index, x in zip(range(0, len(samples)), samples): 
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
     
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in zip(range(0, len(samples)), samples):
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + LikelihoodMoment(x, probabilities[k], 0)
            s1[k] = s1[k] + LikelihoodMoment(x, probabilities[k], 1)
            s2[k] = s2[k] + LikelihoodMoment(x, probabilities[k], 2)
    return s0, s1, s2

def FisherFeatures(annotations, gmm, sc_descriptors):
    vector_len = (2*128+1)*len(gmm[2])
    X,y = imagedataset.GenerateFeatures(FisherVector, annotations, vector_len, gmm, sc_descriptors)
    return X, y

def FisherVectorWeights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def FisherVectorMeans(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def FisherVectorSigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def FisherVector(samples, means, covs, w):
    s0, s1, s2 =  LikelihoodStatistics(samples, means, covs, w)
    T = samples.shape[0]
    a = FisherVectorWeights(s0, s1, s2, means, covs, w, T)
    b = FisherVectorMeans(s0, s1, s2, means, covs, w, T)
    c = FisherVectorSigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv