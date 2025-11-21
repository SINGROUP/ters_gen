import numpy as np
from ters_img_simulator.core.read_gaussian import read_gaussian

def load_molecule(file_name, phi_0, theta_0, psi_0):
    """
    Calculates and returns atomistic Raman polarizabilities of the atoms of the molecule.
    The rotation matrix R used to take into consideration the orientation of the molecule.

    Arguments:
        file_name: Path object -- the Gaussian output file to read
        phi_0, theta_0, psi_0: int -- The desired rotation of the molecule in degrees
    
    Returns: 
        atom_polarizabilities_rotated: np.ndarray -- the atomistic Raman polarizabilities of the molecule
        atoms: int -- the number of atoms in a molecule
        frequencies: np.array -- the vibrational frequencies of each mode (unit: cm^-1)
        atom_pos: np.ndarray -- the cartesian coordinates of the atoms (unit: A)
        R: np.ndarray -- the rotation matrix
    """

    unit_scale = 0.52917721092  # Converting units from bohr to A
    #red_masses, frequencies, polar_derivatives, atom_an, atom_pos, atomic_numbers, N1, N2 = read_gaussian(file_name)
    result = read_gaussian(file_name)
    if result is None:
        return None

    red_masses, frequencies, polar_derivatives, atom_an, atom_pos, atomic_numbers, N1, N2 = result  

    atom_pos = unit_scale*atom_pos.reshape(-1, 3)  # Grouping the positions 
    atoms = N1 + N2  # Number of atoms

    polar_deriv_x = np.zeros((3 * atoms, 3 * atoms))
    polar_deriv_y = np.zeros((3 * atoms, 3 * atoms))
    polar_deriv_z = np.zeros((3 * atoms, 3 * atoms))

    # Separating polarizability derivatives by direction
    for i in range(1, atoms + 1):
        idx = 3 * i - 3
        polar_deriv_x[idx:idx + 3, idx:idx + 3] = [
            [polar_derivatives[6*3*i-18], polar_derivatives[6*3*i-17], polar_derivatives[6*3*i-15]],
            [polar_derivatives[6*3*i-17], polar_derivatives[6*3*i-16], polar_derivatives[6*3*i-14]],
            [polar_derivatives[6*3*i-15], polar_derivatives[6*3*i-14], polar_derivatives[6*3*i-13]]
        ]
        polar_deriv_y[idx:idx + 3, idx:idx + 3] = [
            [polar_derivatives[6*3*i-12], polar_derivatives[6*3*i-11], polar_derivatives[6*3*i-9]],
            [polar_derivatives[6*3*i-11], polar_derivatives[6*3*i-10], polar_derivatives[6*3*i-8]],
            [polar_derivatives[6*3*i-9], polar_derivatives[6*3*i-8], polar_derivatives[6*3*i-7]]
        ]
        polar_deriv_z[idx:idx + 3, idx:idx + 3] = [
            [polar_derivatives[6*3*i-6], polar_derivatives[6*3*i-5], polar_derivatives[6*3*i-3]],
            [polar_derivatives[6*3*i-5], polar_derivatives[6*3*i-4], polar_derivatives[6*3*i-2]],
            [polar_derivatives[6*3*i-3], polar_derivatives[6*3*i-2], polar_derivatives[6*3*i-1]]
        ]

    def sign(num: int):
        if (num > 0):
            return 1
        elif (num < 0):
            return -1
        else:
            return 0

    # Defining the rotation matrix
    phi, theta, psi = phi_0/180*np.pi, theta_0/180*np.pi, psi_0/180*np.pi
    R = np.array([
        [np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi), 
        np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi), 
        np.sin(psi)*np.sin(theta)],
        [-np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi), 
        -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi), 
        np.cos(psi)*np.sin(theta)],
        [np.sin(theta)*np.sin(phi), -np.sin(theta)*np.cos(phi), np.cos(theta)]
    ])

    # R_atom_positions = np.array([
    #     [np.cos(psi), -np.sin(psi), 0],
    #     [np.sin(psi), np.cos(psi), 0],
    #     [0, 0, 1]
    # ])

    # atom_pos_rotated = np.dot(atom_pos, R_atom_positions.T)
    atom_pos_rotated = np.dot(atom_pos, R)

    atom_polarizabilities_rotated = np.zeros((3*atoms-6, 3*atoms, 3*atoms))

    # Calculating atom_polarizabilities and applying rotation to them
    for k in range(1, 3*N1-6*(1-sign(N2))+1):
        atom_amp = atom_an[:, k-1]
        for i in range(1, atoms + 1):
            atom_amp_i = atom_amp[3*i-3:3*i]

            polar_derivative = np.array([
                polar_deriv_x[3*i-3:3*i, 3*i-3:3*i],
                polar_deriv_y[3*i-3:3*i, 3*i-3:3*i],
                polar_deriv_z[3*i-3:3*i, 3*i-3:3*i]
            ])

            atom_polarizabilities = (1/np.sqrt(red_masses[k-1])) * (
                atom_amp_i[0] * polar_derivative[0] +
                atom_amp_i[1] * polar_derivative[1] +
                atom_amp_i[2] * polar_derivative[2]
            )

            polar_atom_rotated_i = np.dot(np.dot(R.T, atom_polarizabilities), R) * unit_scale**2
            atom_polarizabilities_rotated[k-1, 3*i-3:3*i, 3*i-3:3*i] = polar_atom_rotated_i

    return atom_polarizabilities_rotated, atoms, frequencies, atom_pos, atom_pos_rotated, atomic_numbers, R