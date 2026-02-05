
import numpy as np
import matplotlib.pyplot as plt

cmap_colour = 'jet'
save_flag = True

def visualize_3d_molecule(atom_pos, atomic_numbers):
    # Mapping atomic numbers to symbols
    atomic_number_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
        6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne'
    }
    symbols = [atomic_number_to_symbol.get(num, 'X') for num in atomic_numbers]

    # Create XYZ string
    xyz = f"{len(atom_pos)}\nMolecule\n"
    for symbol, pos in zip(symbols, atom_pos):
        xyz += f"{symbol} {pos[0]} {pos[1]} {pos[2]}\n"

    # Set up 3D viewer
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(xyz, 'xyz')
    viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.5}})

    # Add labels for each atom
    for i, (symbol, pos) in enumerate(zip(symbols, atom_pos)):
        viewer.addLabel(symbol, {
            'position': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
            'fontSize': 12,
            'fontColor': 'white',
            'backgroundColor': 'black',
            'alignment': 'center'
        })

    viewer.zoomTo()
    viewer.show()



def plot_channels(mol_image, title_name=None):
    fixed_atomic_numbers = ['background', 'H', 'C', 'N', 'O']
    fig, axes = plt.subplots(1, mol_image.shape[0], figsize=(15, 5))  # Adjust figsize as needed
    for i in range(mol_image.shape[0]):
        ax = axes[i]
        im = ax.imshow(mol_image[i], cmap=cmap_colour)
        ax.set_title(f"Channel {fixed_atomic_numbers[i]}")
        plt.colorbar(im, ax=ax, fraction=0.05)
    if title_name:
        fig.suptitle(title_name, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_flag:
        plt.savefig('save_channels.png', transparent=True)
    plt.show()



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


def remove_disk(atom_pos, atomic_numbers):
    # Convert all positions to a numpy array
    ref = 1
    positions = np.array(atom_pos)
    zmax = np.max(positions[:, 2])
    positions = positions[positions[:, 2] > zmax - ref]

    filtered_atomic_numbers = [num for num, pos in zip(atomic_numbers, atom_pos) if pos[2] > (zmax - ref)]
    filtered_positions = [pos for pos in atom_pos if pos[2] > (zmax - ref)]

    '''
    print("Initial", filtered_positions)

    # Normalize the atom_pos by maximum distance
    pos = np.array(filtered_positions)
    cov = pos.T @ pos
    #filtered_positions /= np.max(cov)

    print("Normalized", filtered_positions)
    '''

    return filtered_positions, filtered_atomic_numbers


def molecule_circular_image(atom_pos, atomic_numbers, circle_radius=3):
    """
    Generate a multi-channel circular image representation of a molecule.
    
    Parameters:
      atom_pos: List or array of [x, y, z] coordinates for each atom.
      atomic_numbers: List of atomic numbers (integers) for each atom.
      circle_radius: Radius of the circle to draw for each atom in pixels.
      
    Returns:
      multi_channel_img: A numpy array of shape (num_elements, grid_size, grid_size).
    """
    # Define fixed channels for these atomic numbers
    fixed_atomic_numbers = [1, 6, 7, 8]  # H, C, N, O
    grid_size = 64  # Grid resolution
    
    filtered_positions, filtered_atomic_numbers = remove_disk(atom_pos, atomic_numbers)
    positions = np.array(filtered_positions)
    # Calculate the center of the molecule (using filtered positions)
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1])
    
    # Set fixed grid size of 18 Å centered on the molecule
    grid_physical_size = 18.0  # Å
    x_min = center_x - grid_physical_size / 2  # -9 Å
    x_max = center_x + grid_physical_size / 2  # +9 Å
    y_min = center_y - grid_physical_size / 2  # -9 Å
    y_max = center_y + grid_physical_size / 2  # +9 Å
    
    # Initialize multi-channel image array
    multi_channel_img = np.zeros((len(fixed_atomic_numbers), grid_size, grid_size))
    
    for ch, atomic_num in enumerate(fixed_atomic_numbers):
        # Extract positions for atoms that match the current atomic number
        pos = np.array([pos for num, pos in zip(filtered_atomic_numbers, filtered_positions) if num == atomic_num])
        if pos.size == 0:
            continue  # Skip if no atoms of this atomic number
        # Use only x and y coordinates
        pos = pos[:, :2]
    
        # Convert physical coordinates to grid indices
        x_idx = np.clip(((pos[:, 0] - x_min) / (x_max - x_min) * grid_size).astype(int), 0, grid_size - 1)
        y_idx = np.clip(((pos[:, 1] - y_min) / (y_max - y_min) * grid_size).astype(int), 0, grid_size - 1)
    
        # For each atom, add a circular disk to the image
        for x, y in zip(x_idx, y_idx):
            # Note: image indexing is (row, column) so we use (y, x)
            add_disk(multi_channel_img[ch], (y, x), circle_radius)
    
    return multi_channel_img



def normalize(spectrums):
    mean = np.mean(spectrums, axis = (0,1))
    std = np.std(spectrums, axis=(0,1))
    std[std==0] = 1
    spectrums = (spectrums - mean)/std
    return spectrums

def minmax(spectrums):
    min = np.min(spectrums, axis = (0,1))
    max = np.max(spectrums, axis = (0,1))
    diff = max - min 
    diff[diff==0] = 1
    spectrums = (spectrums - min)/diff

    return spectrums

def uniform_channels(spectrums, frequencies):
    """
    Create uniform channels from the spectrums.
    Returns an array of shape (64, 64, 40) by averaging spectrums in frequency bins.
    """
    # Initialize output array: 40 channels, each 64x64
    num_channels = 400
    shape = spectrums.shape[0]
    channels = np.zeros((shape, shape, num_channels))
    max_freq = 4000
    
    # Define step size based on the maximum frequency and number of channels.
    step = max_freq // num_channels

    spectrums = minmax(spectrums)

    count = 0 
    # For each frequency bin, compute the average spectrum.
    for i in range(1, max_freq, step):
        
        # Find indices for frequencies in this bin. Adjust band width as needed.
        indices = (frequencies > i) & (frequencies < i + step)
        selected_spectrums = spectrums[:, :, indices]
        # If no data or the data is all zeros, continue.
        if selected_spectrums.size == 0 or np.all(selected_spectrums == 0):
            count += 1
            continue
        # Compute the mean across the frequency axis
        channels[:, :, count] = np.mean(selected_spectrums, axis=2)
        count += 1

    return channels


def get_element_color(atomic_number):
    """Returns a color based on atomic number."""
    # Map atomic numbers to colors (same as original but keyed by number)
    color_map = {
        1: 'lightgray',  # Hydrogen
        2: 'cyan',       # Helium
        3: 'purple',     # Lithium
        4: 'brown',      # Beryllium
        5: 'pink',       # Boron
        6: 'black',      # Carbon
        7: 'blue',       # Nitrogen
        8: 'red',        # Oxygen
        9: 'green',      # Fluorine
        10: 'orange'     # Neon
    }
    return color_map.get(atomic_number, 'gray')  # Default color for unknown elements



def overlay_molecule_on_image(atom_pos, atomic_numbers, image, axes, grid_size=18.0, bond_threshold=1.6):
    """
    Overlay molecule visualization on a given image (e.g., average channel or individual channel).
    
    Parameters:
    - atom_pos: Array of [x, y, z] coordinates for each atom (in angstroms).
    - atomic_numbers: List of atomic numbers.
    - image: 2D numpy array (64x64) to overlay the molecule on.
    - axes: Matplotlib axes object to plot on.
    - grid_size: Physical size of the grid in angstroms (default 16 Å).
    - bond_threshold: Maximum distance for drawing bonds (in angstroms).
    """
    # Display the image as the background
    axes.imshow(image, cmap=cmap_colour, extent=(-grid_size/2, grid_size/2, -grid_size/2, grid_size/2), origin='lower')

    # Center the molecule
    positions = np.array(atom_pos)
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1])
    centered_pos = positions - [center_x, center_y, 0]  # Center in x-y plane
    print(center_x, center_y)
    centered_pos = positions

    # Draw bonds
    num_atoms = len(centered_pos)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(centered_pos[i] - centered_pos[j])
            if distance < bond_threshold:
                axes.plot([centered_pos[i, 0], centered_pos[j, 0]],
                          [centered_pos[i, 1], centered_pos[j, 1]],
                          'gray', linewidth=1.5, zorder=1)

    # Draw atoms
    for i, (x, y, _) in enumerate(centered_pos):
        color = get_element_color(atomic_numbers[i])
        symbol = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
                  6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne'}.get(atomic_numbers[i], 'X')
        axes.scatter(x, y, s=200, c=color, edgecolors='black', zorder=2)
        axes.text(x, y, symbol, fontsize=10, color='white',
                  ha='center', va='center', fontweight='bold', zorder=3)

    # Set axis properties
    axes.set_xlim(-grid_size/2, grid_size/2)
    axes.set_ylim(-grid_size/2, grid_size/2)
    axes.set_aspect('equal')
    axes.set_axis_off()

    if save_flag:
        axes.figure.savefig('plot_overlay.png', transparent=True, bbox_inches='tight', pad_inches=0.1)



def molecule_visualization_image(atom_pos, atomic_numbers, bond_threshold=1.6, axes=None):
    """
    Creates a scatter plot visualization of a molecule with element-based colors and bonds.
    
    Parameters:
    - atom_pos: List or array of [x, y, z] coordinates for each atom.
    - atomic_numbers: List of atomic numbers (integers) for each atom.
    - bond_threshold: Maximum distance (in units of atom_pos) for drawing bonds.
    - axes: Matplotlib axes object to plot on. If None, a new figure and axes are created.
    """
    # Define mapping from atomic numbers to element symbols
    atomic_number_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
        6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne'
        # Add more elements as needed
    }

    # Convert atomic numbers to symbols, using 'X' for unknown elements
    element_symbols = [atomic_number_to_symbol.get(num, 'X') for num in atomic_numbers]

    # Set up the axes
    if axes is None:
        fig, axes = plt.subplots(figsize=(5, 5))
        created_figure = True
    else:
        created_figure = False

    # Draw bonds based on distance threshold
    num_atoms = len(atom_pos)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if distance < bond_threshold:
                axes.plot([atom_pos[i, 0], atom_pos[j, 0]],
                          [atom_pos[i, 1], atom_pos[j, 1]],
                          'gray', linewidth=1.5, zorder=1)

    # Draw atoms with different colors
    for i, (x, y, _) in enumerate(atom_pos):
        atomic_number = atomic_numbers[i]
        color = get_element_color(atomic_number)
        symbol = element_symbols[i]
        axes.scatter(x, y, s=200, c=color, edgecolors='black', zorder=2)
        axes.text(x, y, symbol, fontsize=10, color='white',
                  ha='center', va='center', fontweight='bold')

    # Create legend manually
    unique_atomic_numbers = set(atomic_numbers)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                          markerfacecolor=get_element_color(num),
                          markeredgecolor='black',
                          label=atomic_number_to_symbol.get(num, 'X'))
               for num in unique_atomic_numbers]
    #axes.legend(handles=handles, title="Elements", loc="upper right")

    # Set title and axis properties
    #axes.set_title("Molecule Visualization")
    axes.set_aspect('equal')
    axes.set_axis_off()

    # Finalize layout and display if a new figure was created
    if created_figure:
        fig.tight_layout()
        plt.show()

    

        