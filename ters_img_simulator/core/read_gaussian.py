import numpy as np

def read_gaussian(file_name):
    """
    Read a Gaussian output file and return at least the vibrational frequencies,
    polarizabilities, and polarizability derivatives.

    Arguments:
        file_name: Path object -- the name of the Gaussian output file to read

    Returns:
        red_masses: np.ndarray -- the reduced masses
        frequencies: np.ndarray -- the vibrational frequencies of each mode
        polar_derivatives: np.ndarray -- the polarizability derivatives
        atom_an: np.ndarray -- ???
        atom_positions -- the cartesian coordinates of the atoms
        N1: int -- number of free atoms in a molecule
        N2: int -- number of fixed atoms in a molecule
    """

    def read_frequencies_red_masses(N1, N2, file):
        """
        Reads and returns vibrational frequencies of each mode and reduced masses.
        
        Arguments:
            N1: int -- the number of free atoms of a molecule
            N2: int -- the number of fixed atoms of a molecule

        Returns:
            frequencies: np.ndarray
            red_masses: np.ndarray
        """
        file.seek(0)
        label = "Vib-E2"
        if locate_label(label, file) == None:
            return None
        
        line = file.readline()
        line_data = [float(element) for element in line.split()]
        if (N2 == 0):
            frequencies, current_line, current_element = read_n_elements(3*(N1-2), 0, line_data, file)
            red_masses, _, _ = read_n_elements(3*(N1-2), current_element, current_line, file)
        else:
            frequencies, current_line, current_element = read_n_elements(3*N1, 0, line_data, file)
            red_masses, _, _ = read_n_elements(3*N1, current_element, current_line, file)
        
        return frequencies, red_masses
 

    def read_atom_an(N1, N2, file):
        """
        Reads and returns Vib-Modes.

        Returns:
            atom_an: np.ndarray
        """
        file.seek(0)
        label = "Vib-Modes"
        if locate_label(label, file) == None:
            return None

        line = file.readline()
        line_data = [float(element) for element in line.split()]
        if (N2 == 0):
            atom_an, _, _ = read_n_elements(3*N1*3*(N1-2), 0, line_data, file)
            atom_an = atom_an.reshape((3*(N1-2), 3*N1))
            atom_an = atom_an.T
        else:
            atom_an, _, _ = read_n_elements(3*N1*3*(N1-2), 0, line_data, file)
            atom_an = atom_an.reshape((3*N1, 3*(N1+N2)))
            atom_an = atom_an.T

        return atom_an


    def read_polar_derivatives(N1, N2, file):
        """
        Reads and returns polarizability derivatives.

        Returns:
            polar_derivatives: np.ndarray
        """
        file.seek(0)
        label = "Polarizability Derivatives"
        if locate_label(label, file) == None:
            return None
        
        line = file.readline()
        line_data = [float(element) for element in line.split()]
        polar_derivatives, _, _ = read_n_elements(6*3*(N1+N2), 0, line_data, file)

        return polar_derivatives


    def read_atom_positions(N1, N2, file):
        """
        Reads and returns coordinates of the atoms.

        Returns:
            atom_positions: np.ndarray
        """
        file.seek(0)
        label = "Current cartesian coordinates"
        if locate_label(label, file) == None:
            return None
        
        line = file.readline()
        line_data = [float(element) for element in line.split()]
        atom_positions, _, _ = read_n_elements(3*(N1+N2), 0, line_data, file)

        return atom_positions
    

    def read_atomic_numbers(N1, N2, file):
        """
        Reads and returns atomic numbers of the atoms.

        Returns:
            atom_an: np.ndarray
        """
        file.seek(0)
        label = "Atomic numbers"
        if locate_label(label, file) == None:
            return None
        
        line = file.readline()
        line_data = [int(element) for element in line.split()]
        atomic_numbers, _, _ = read_n_elements(N1+N2, 0, line_data, file)
        atomic_numbers = atomic_numbers.astype(int)

        return atomic_numbers


    def read_molecule_constituent(file):
        """
        Reads and returns the number of fixed (N2) and not free atoms in a molecule (N1).

        Returns:
            N1: int
            N2: int
        """
        file.seek(0)
        label = "MicOpt"
        N1, N2 = 0, 0
        if locate_label(label, file) == None:
            return None
            
        line = file.readline()
        while True:
            S1 = line.count("-1")
            S2 = line.count("-2")
            N1 = N1 + S1
            N2 = N2 + S2
            if (S1 + S2 < 6):
                break
            line = file.readline()
        
        return N1, N2


    def read_n_elements(n: int, start: int, current_line_data,  file):
        """
        Reads the next n elements in the Gaussian output file. current_line_data consists of float values
        that have been read from current line in the output file. start is the index from which the
        reading is continued in current_line_data.

        Returns:
            data: np.ndarray -- contains the n read elements of the Gaussian output file
            line_data: List[float] -- the data of the line where reading stopped
            current_element: int -- the first index of line_data that was left unread
        """
        data = []
        count = 0
        # Collecting the elements from the current row
        for i in range(len(current_line_data) - start):
            if count < n:
                data.append(current_line_data[start+i])
                count += 1
            else:
                return np.array(data), current_line_data, start + count
        
        # Check if collected enough
        if (count >= n):
            return np.array(data), current_line_data, start + count

        # Collecting the elements after the first row
        exit_loop = False
        line_data = []
        current_element = 0
        while not exit_loop:
            line = file.readline()
            line_data = [float(element) for element in line.split()]
            for i in range(len(line_data)):
                current_element = i
                data.append(line_data[i])
                count += 1
                if (count >= n):
                    exit_loop = True
                    break
        data = np.array(data)
        
        return data, line_data, current_element + 1


    def locate_label(label: str, file):
        """
        Moves the file pointer to the line where given label exists in the Gaussian output file.
        """
        for line in file:
            if label in line:
                return line
        return None

    # Reading the data from the Gaussian ouput file
    with open(file_name, 'r') as file:
        #N1, N2 = read_molecule_constituent(file)
        N1_N2 = read_molecule_constituent(file)
        if N1_N2 is None:
            return None
        N1, N2 = N1_N2

        #frequencies, red_masses = read_frequencies_red_masses(N1, N2, file)
        frequencies_red_masses = read_frequencies_red_masses(N1, N2, file)
        atom_an = read_atom_an(N1, N2, file)
        polar_derivatives = read_polar_derivatives(N1, N2, file)
        atom_positions = read_atom_positions(N1, N2, file)
        atomic_numbers = read_atomic_numbers(N1, N2, file)
    
    parameters = [frequencies_red_masses, polar_derivatives, atom_an, atom_positions, atomic_numbers, N1, N2]
    if any(parameter is None for parameter in parameters):
        return None
    frequencies, red_masses = frequencies_red_masses

    return red_masses, frequencies, polar_derivatives, atom_an, atom_positions, atomic_numbers, N1, N2