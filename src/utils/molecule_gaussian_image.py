import numpy as np
from scipy.ndimage import gaussian_filter

def molecule_gaussian_image(xyz_string):

    lines = xyz_string.strip().split("\n")[2:]  # Skip first two lines (comment and atom count)
    
    elements = []
    positions = []
    atoms_data = []  # This will store tuples of (element, (x, y, z))
    
    for line in lines:
        parts = line.split()
        element = parts[0]  # First entry is the element symbol
        pos = tuple(float(x) for x in parts[1:4])  # Next three entries are x, y, z coordinates
        elements.append(element)
        positions.append(pos)
        atoms_data.append((element, pos))
    
    positions = np.array(positions)


    fixed_elements = ["H", "C", "N", "O"]

    # Define grid parameters
    grid_size = 256  # Grid resolution (same for all channels)
    sigma = 1.5  # Gaussian blur intensity (adjustable)
    
    # Assume 'positions' is a numpy array of atom positions (shape: [n_atoms, 3])
    # and 'atoms_data' is a list of tuples like (element, (x, y, z))
    positions_array = np.array(positions)  # positions should come from your parsed data
    
    # Determine grid boundaries using only x and y coordinates, with a margin
    x_min, y_min = positions_array[:, :2].min(axis=0) - 1.0
    x_max, y_max = positions_array[:, :2].max(axis=0) + 1.0
    x_lin = np.linspace(x_min, x_max, grid_size)
    y_lin = np.linspace(y_min, y_max, grid_size)
    
    # Initialize the multi-channel image array with fixed channels
    multi_channel_img = np.zeros((len(fixed_elements), grid_size, grid_size))
    
    # Map atomic positions to grid for each fixed channel
    for ch, elem in enumerate(fixed_elements):
        # Extract positions for atoms that match the fixed element
        pos = np.array([pos for e, pos in atoms_data if e == elem])
        if pos.size == 0:
            continue  # If no atoms of this element, skip to next channel
        pos = pos[:, :2]  # Use only x and y coordinates
    
        # Convert physical coordinates to grid indices
        x_idx = np.clip(((pos[:, 0] - x_min) / (x_max - x_min) * grid_size).astype(int), 0, grid_size - 1)
        y_idx = np.clip(((pos[:, 1] - y_min) / (y_max - y_min) * grid_size).astype(int), 0, grid_size - 1)
    
        # Mark atom positions in the grid
        for x, y in zip(x_idx, y_idx):
            multi_channel_img[ch, y, x] = 1
    
        # Apply Gaussian smoothing to spread the atomic presence
        multi_channel_img[ch] = gaussian_filter(multi_channel_img[ch], sigma=sigma)

    return multi_channel_img
    