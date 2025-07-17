import numpy as np

def add_disk(image, center, radius):
    """
    Draw a disk (circle) on a 2D image.
    
    Parameters:
      image: 2D numpy array where the disk will be added.
      center: Tuple (row, col) indicating the center of the disk.
      radius: Radius of the disk in pixels.
    """
    rows, cols = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (rows - center[0])**2 + (cols - center[1])**2 <= radius**2
    image[mask] = 1  # You can adjust the intensity if needed

def molecule_circular_image(xyz_string, flag=True, circle_radius=5):
    lines = xyz_string.strip().split("\n")[2:]  # Skip header lines
    
    atoms_data = []  # Store tuples of (element, (x, y, z))
    
    for line in lines:
        parts = line.split()
        element = parts[0]  # First entry is the element symbol
        pos = tuple(float(x) for x in parts[1:4])
        atoms_data.append((element, pos))
    
    # Define fixed channels for these elements
    fixed_elements = ["H", "C", "N", "O"]
    grid_size = 256  # Grid resolution
    
    # Convert all positions to a numpy array
    positions = np.array([pos for _, pos in atoms_data])
    zmax = np.max(positions[:, 2])
    positions = positions[positions[:, 2] > zmax - 1.0]
    
    atoms_data = [(e, pos) for e, pos in atoms_data if pos[2] > (zmax - 1.0)]
    
    # Calculate the center of the molecule (using filtered positions)
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1])
    
    # Set fixed grid size of 18 Å centered on the molecule
    grid_physical_size = 18.0  # Å
    x_min = center_x - grid_physical_size / 2  # -9 Å
    x_max = center_x + grid_physical_size / 2  # +9 Å
    y_min = center_y - grid_physical_size / 2  # -9 Å
    y_max = center_y + grid_physical_size / 2  # +9 Å
    
    # Create grid linspaces (optional, not used directly but kept for consistency)
    x_lin = np.linspace(x_min, x_max, grid_size)
    y_lin = np.linspace(y_min, y_max, grid_size)
    
    # Initialize multi-channel image array
    if flag:
        multi_channel_img = np.zeros((1, grid_size, grid_size))
    else:
        multi_channel_img = np.zeros((len(fixed_elements)+1, grid_size, grid_size))
    #multi_channel_img = np.zeros((1, grid_size, grid_size))
    
    for ch, elem in enumerate(fixed_elements):
        # Extract positions for atoms that match the current element
        pos = np.array([pos for e, pos in atoms_data if e == elem])
        if pos.size == 0:
            continue  # Skip if no atoms of this element
        # Use only x and y coordinates
        pos = pos[:, :2]
    
        # Convert physical coordinates to grid indices
        x_idx = np.clip(((pos[:, 0] - x_min) / (x_max - x_min) * grid_size).astype(int), 0, grid_size - 1)
        y_idx = np.clip(((pos[:, 1] - y_min) / (y_max - y_min) * grid_size).astype(int), 0, grid_size - 1)
    
        # For each atom, add a circular disk to the image
        for x, y in zip(x_idx, y_idx):
            # Note: image indexing is (row, column) so we use (y, x)
            if flag:
                ch = 0
            add_disk(multi_channel_img[ch], (y, x), circle_radius)

    # Add background as the last channel if not flag
    if not flag:
        foreground = np.clip(np.sum(multi_channel_img[:-1], axis=0), 0, 1)
        multi_channel_img[-1] = 1 - foreground
    
    return multi_channel_img


