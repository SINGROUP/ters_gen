import numpy as np

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


def read_npz(npz_file):
    with np.load(npz_file) as data:
        atom_pos = data['atom_pos']
        atomic_numbers = data['atomic_numbers']
    return atom_pos, atomic_numbers