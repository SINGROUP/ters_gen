# import all packages
import numpy as np



#Helper function packages
import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


def read_npz(npz_file):
    
    # Load the .npz file and extract data, as per your example
    try:
        with np.load(npz_file) as data:
            atom_pos = data['atom_pos']
            # You can also access these if needed later:
            atomic_numbers = data['atomic_numbers']
            # frequencies = data['frequencies']
            # spectrums = data['spectrums']
            
    except Exception as e:
        print(f"Error processing {os.path.basename(npz_file)}: {e}")

    return atom_pos, atomic_numbers


def read_fchk(fchk_file):
    atomic_numbers = []
    coordinates = []
    num_atoms = 0

    with open(fchk_file, 'r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('Number of atoms'):
                num_atoms = int(line.split()[-1])
            elif line.startswith('Atomic numbers'):
                start_line = i + 1
                values = []
                while len(values) < num_atoms and start_line < len(lines):
                    values.extend([int(x) for x in lines[start_line].split()])
                    start_line += 1
                atomic_numbers = values[:num_atoms]
            elif line.startswith('Current cartesian coordinates'):
                start_line = i + 1
                values = []
                while len(values) < 3 * num_atoms and start_line < len(lines):
                    values.extend([float(x) for x in lines[start_line].split()])
                    start_line += 1
                coordinates = values[:3 * num_atoms]
    
    
    coordinates = np.array(coordinates).reshape(num_atoms, 3)
    #xyz = np.column_stack((coordinates, atomic_numbers))
    return coordinates, atomic_numbers



def pca(coords):
    

    N = coords.shape[0]
    centroid = coords.mean(axis = 0)
    X = coords - centroid # center

    # Center and PCA
    C = (X.T@X)/N # covariance matrix
    eigvals, eigvecs = np.linalg.eigh(C)

    return eigvals, eigvecs, X



# Compute rotated coords lying in the z plane
def rotation(normal, X):
    # compute rotation
    # want n is k = [0, 0, 1]
    k = np.array([0,0,1.0])
    u = np.cross(normal, k)
    theta = np.arccos(np.dot(normal, k))

    # Rotate so that the plane z = c
    # Rodrigues' rotation matrix

    # Numerical noise
    if np.linalg.norm(u) < 1e-8 or theta == 0:
        return X, np.eye(3)
    
    u /= np.linalg.norm(u)

    K = np.array([[    0,   -u[2],  u[1]],
                  [ u[2],     0,  -u[0]],
                  [-u[1],  u[0],     0]])
    R = np.eye(3)*np.cos(theta) + np.sin(theta)*K + (1-np.cos(theta))*(u[:,None]@u[None,:])

    rotated_coords = (R@X.T).T # apply to each point

    return rotated_coords, R


def rotation_to_zyz_euler(R, eps = 1e-8):

    # Sanity checks
    if not np.allclose(R@R.T, np.eye(3), atol=1e-6):
        raise ValueError("R is not orthogonal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("R must have det = +1")
    
    # Compute beta = angle about Y
    beta = np.arccos(R[2,2])

    # Generic case: sin(beta) != 0
    if abs(np.sin(beta)) > eps:
        alpha = np.arctan2( R[1, 2], R[0, 2] )
        gamma = np.arctan2( R[2, 1], -R[2, 0] )
    else:
        # Gimbal lock (beta ~ 0 or pi): only α+γ is determined
        # here we choose γ = 0 and absorb everything into α
        gamma = 0.0
        # when beta ≈ 0, R ≈ Rz(α+γ), so:
        alpha = np.arctan2(R[1, 0], R[0, 0])

    return alpha, beta, gamma


def zyz_rotation(alpha, beta, gamma):
    """
    Construct the 3×3 rotation matrix for Z–Y–Z Euler angles.
    
    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    """
    # Rotation about Z by alpha
    Rz1 = np.array([
        [ np.cos(alpha), -np.sin(alpha), 0],
        [ np.sin(alpha),  np.cos(alpha), 0],
        [             0,              0, 1]
    ])

    # Rotation about Y by beta
    Ry = np.array([
        [  np.cos(beta), 0, np.sin(beta)],
        [             0, 1,            0],
        [ -np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation about Z by gamma
    Rz2 = np.array([
        [ np.cos(gamma), -np.sin(gamma), 0],
        [ np.sin(gamma),  np.cos(gamma), 0],
        [              0,               0, 1]
    ])

    return Rz1 @ Ry @ Rz2




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


def normal_to_zyz_ters(normal):
    """
    Turn a 3-vector `normal` into angles (φ₀, θ₀, ψ₀) in degrees
    so that rotation_zyz_ters(φ₀,θ₀,ψ₀,…) carries normal → +Z.
    """
    n = normal/np.linalg.norm(normal)
    phi   = -np.degrees(np.arctan2(n[1], n[0]))
    theta =  np.degrees(np.arccos(   n[2]    ))
    psi   =  0
    
    return phi, theta, psi