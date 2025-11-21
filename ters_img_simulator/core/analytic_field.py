import numpy as np

def analytic_field(atom_polarizabilities, atoms, frequencies, E_type, atom_pos=None, R=None, tip_xyz=None, tip_width=None):
    """
    Calculates and returns intensities of each vibrational mode and polarized dipoles of each 
    atom for each vibrational mode.

    Arguments:
    atom_polarizabilities: np.ndarray -- the polarizability of each atom
    atoms: int -- the number of atoms in the molecule
    frequencies: np.ndarray -- the vibrational frequencies of each mode (unit: cm^-1)
    E_type: int -- type of the incident field (0: plane wave, 1: 2D Gaussian, 2: 3D Gaussian)
    atom_pos: np.ndarray -- the cartesian coordinates of the atoms (unit: A)
    R: np.ndarray -- the rotation matrix
    tip_xyz: np.ndarray -- position of the tip (unit: A)
    tip_width: np.ndarray -- width of the Gaussian field (unit: A)
    
    Returns:
        intensity: np.ndarray -- intensity of each vibrational mode
        dipole: np.ndarray -- polarized dipole of each atom for each mode
    """
    E_local = np.zeros(3*atoms)

    match E_type:
        case 0: # plane wave
            theta = 0/180*np.pi #Angle between incident direction and z-axis
            phi = 0/180*np.pi   #Angle of incident direction around z-axis
            for i in range(1, atoms + 1):
                E_local[3*i-3:3*i] = \
                    np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

        case 1: # 2D Gaussian
            for i in range(1, atoms + 1):
                atom_tip_distance = np.dot(atom_pos[i-1, :], R) - tip_xyz
                E_local[3*i-3:3*i] = \
                    np.exp(-4*np.log(2)*np.sum(((atom_tip_distance[0:2])/(tip_width[0:2]))**2))*np.array([0, 0, 1])

        case 2: # 3D Gaussian
            for i in range(1, atoms + 1):
                atom_tip_distance = np.dot(atom_pos[i-1, :], R) - tip_xyz
                E_local[3*i-3:3*i] = \
                    np.exp(-4*np.log(2)*np.sum(((atom_tip_distance)/(tip_width))**2))*(atom_tip_distance/np.linalg.norm(atom_tip_distance))

        case _:
            raise ValueError(f"Function analytic field only supports field types 0, 1, 2. Input values was {E_type}")

    # Calculating mode intensities and dipoles
    mode_intensities = np.zeros((3*atoms-6, 2))
    dipoles = np.tensordot(atom_polarizabilities, E_local, axes=([2], [0]))
    E_scattering = np.einsum('i,ij->j', E_local, dipoles.T)

    mode_intensities[:, 0] = frequencies
    mode_intensities[:, 1] = E_scattering**2

    return mode_intensities, dipoles