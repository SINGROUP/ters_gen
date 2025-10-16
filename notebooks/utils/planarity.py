# import all packages
import numpy as np



#Helper function packages
import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt



def planarity(eigvals, eigvecs, X):
    # sort descending 
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]


    # PCA-based planarity
    planarity_pca = 100*(eigvals[0] + eigvals[1])/eigvals.sum()
    


    # RMSD based planarity
    normal = eigvecs[:, 2] # eigenvector for smallest eigenvalue
    d = (X@normal)/np.linalg.norm(normal)
    rmsd = np.sqrt(np.mean(d**2))
    L = np.sqrt(eigvals[0] + eigvals[1])
    planarity_rms = 100*(1-rmsd/L)
    

    return planarity_pca, planarity_rms, rmsd


def pca(coords):
    

    N = coords.shape[0]
    centroid = coords.mean(axis = 0)
    X = coords - centroid # center

    # Center and PCA
    C = (X.T@X)/N # covariance matrix
    eigvals, eigvecs = np.linalg.eigh(C)

    return eigvals, eigvecs, X
