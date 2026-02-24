'''
The particular script gives various orientation to increase the diversity of orientations for training the model. 
However, initially, I am just interested in looking at the planar molecule. '''


import numpy as np
from scipy.spatial import ConvexHull
import scipy

def parse_fchk(file_path):
    """
    Extract atomic numbers and Cartesian coordinates from a Gaussian .fchk file.
    
    Args:
        file_path (str): Path to the .fchk file.
    
    Returns:
        np.ndarray: Array of shape (n, 4) with [x, y, z, atomic_number] for each atom.
    """
    atomic_numbers = []
    coordinates = []
    num_atoms = 0
    
    with open(file_path, 'r') as f:
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
    
    if not atomic_numbers or not coordinates or num_atoms == 0:
        raise ValueError("Failed to parse atomic numbers or coordinates from .fchk file.")
    
    coordinates = np.array(coordinates).reshape(num_atoms, 3)
    xyz = np.column_stack((coordinates, atomic_numbers))
    return xyz

def _convert_elemements(bias_dict):
    """Convert element symbols to atomic numbers or leave as is."""
    elem_map = {
        'H': 1,   'He': 2,  'Li': 3,  'Be': 4,  'B': 5,
        'C': 6,   'N': 7,   'O': 8,   'F': 9,   'Ne': 10
    }
    return {elem_map.get(k, k): v for k, v in bias_dict.items()}

def plane_from_3points(points):
    """Calculate plane equation from 3 points: ax + by + cz + d = 0."""
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, p1)
    return np.array([*normal, d])

def plane_from_2points(points):
    """Calculate plane equation from 2 points, assuming z=0 for third direction."""
    p1, p2 = points
    v = p2 - p1
    if abs(v[0]) > abs(v[1]):
        u = np.array([0, 1, 0])
    else:
        u = np.array([1, 0, 0])
    normal = np.cross(v, u)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, p1)
    return np.array([*normal, d])

def get_convex_hull_eqs(xyz, angle_tolerance=5):
    """Get plane equations from convex hull of molecular coordinates."""
    points = xyz[:, :3]
    try:
        hull = ConvexHull(points)
        eqs = []
        for simplex in hull.simplices:
            pts = points[simplex]
            eq = plane_from_3points(pts)
            eqs.append(eq)
        return np.array(eqs), hull
    except scipy.spatial.qhull.QhullError:
        raise

def find_planar_segments(xyz, eqs, dist_tol=0.1, num_atoms=6):
    """Identify planar segments with at least num_atoms within dist_tol."""

    """Need to adapt this code to detect all the planes which have more than 3 atoms, and sort them according to the most elements"""
    planar_eqs = []
    planar_indices = []
    coords = xyz[:, :3]
    for i, eq in enumerate(eqs):
        normal, d = eq[:3], eq[3]
        distances = np.abs(np.dot(coords, normal) + d) / np.linalg.norm(normal)
        in_plane = np.where(distances < dist_tol)[0]
        if len(in_plane) >= num_atoms:
            planar_eqs.append(eq)
            planar_indices.append(i)
    return np.array(planar_eqs), np.array(planar_indices)

def get_plane_elements(xyz, eqs, dist_tol=0.7):
    """Get elements within dist_tol of each plane."""
    coords = xyz[:, :3]
    elements = xyz[:, -1].astype(int)
    plane_elems = []
    for eq in eqs:
        normal, d = eq[:3], eq[3]
        distances = np.abs(np.dot(coords, normal) + d) / np.linalg.norm(normal)
        in_plane = np.where(distances < dist_tol)[0]
        plane_elems.append(set(elements[in_plane]))
    return plane_elems

def random_unit_vector():
    """Generate a random unit vector in 3D."""
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def cart_to_sph(vec):
    """Convert Cartesian vector to spherical coordinates (r, phi, theta)."""
    x, y, z = vec
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r) if r > 0 else 0
    return r, phi, theta

def zyz_rotation(phi, theta, psi):
    """Generate ZYZ Euler rotation matrix."""
    c1, s1 = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(psi), np.sin(psi)
    R1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    R2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    R3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    return R1 @ R2 @ R3

def matrix_to_zyz_angles(R, deg=True):
    """
    Convert a 3x3 rotation matrix to ZYZ Euler angles (phi, theta, psi).
    
    Args:
        R (np.ndarray): 3x3 rotation matrix.
        deg (bool): If True, return angles in degrees; else radians.
    
    Returns:
        tuple: (phi, theta, psi) angles.
    """
    R = np.asarray(R)
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    
    # Extract theta
    theta = np.arccos(np.clip(R[2, 2], -1, 1))
    
    # Check for singularity (theta = 0 or pi)
    if np.abs(np.sin(theta)) < 1e-10:
        # When theta = 0 or pi, phi and psi combine
        phi = 0
        psi = np.arctan2(R[1, 0], R[0, 0])
    else:
        phi = np.arctan2(R[1, 2], R[0, 2])
        psi = np.arctan2(R[2, 1], -R[2, 0])
    
    if deg:
        phi = np.degrees(phi)
        theta = np.degrees(theta)
        psi = np.degrees(psi)
    
    return phi, theta, psi

def apply_zyz_rotation(xyz, phi, theta, psi, deg=True):
    """
    Apply ZYZ Euler rotation to xyz coordinates.
    
    Args:
        xyz (np.ndarray): Array of shape (n, 3) or (n, 4) with coordinates.
        phi, theta, psi (float): ZYZ Euler angles.
        deg (bool): If True, angles are in degrees; else radians.
    
    Returns:
        np.ndarray: Rotated coordinates.
    """
    if deg:
        phi, theta, psi = np.radians([phi, theta, psi])
    R = zyz_rotation(phi, theta, psi)
    coords = xyz[:, :3] if xyz.shape[1] > 3 else xyz
    rotated = coords @ R
    if xyz.shape[1] > 3:
        return np.column_stack((rotated, xyz[:, 3]))
    return rotated

def choose_rotations_bias(xyz, flat=True, plane_bias={'O': 1, 'C': 0.8, 'H': 0.5},
                         random_bias={'O': 0.5, 'C': 0.4, 'H': 2}, angle_tolerance=5,
                         elem_dist_tol=0.7, flat_dist_tol=0.1, flat_num_atoms=6):
    """
    Choose rotation matrices and angles for a molecule based on geometry and element biases.
    
    Args:
        xyz (np.ndarray): Array of shape (n, 4) with x, y, z coordinates and element IDs.
        flat (bool): If True, prioritize planar segments.
        plane_bias (dict): Probability of selecting planes containing specific elements.
        random_bias (dict): Number of random rotations biased towards elements.
        angle_tolerance (float): Minimum angle (degrees) between normal vectors.
        elem_dist_tol (float): Distance tolerance for elements in a plane.
        flat_dist_tol (float): Distance tolerance for planar segments.
        flat_num_atoms (int): Minimum number of atoms in a planar segment.
    
    Returns:
        list: List of (rotation_matrix, angles, source) tuples, where angles are (phi, theta, psi).
    """
    n_vecs = []
    sources = []

    plane_bias = _convert_elemements(plane_bias)
    random_bias = _convert_elemements(random_bias)
    
    if len(xyz) > 3:
        try:
            eqs, hull = get_convex_hull_eqs(xyz, angle_tolerance=angle_tolerance)
            vertices = hull.vertices
        except scipy.spatial.qhull.QhullError:
            print(f'A problematic molecule encountered.')
            return []
    elif len(xyz) == 3:
        eqs = plane_from_3points(xyz[:, :3])[None]
        vertices = np.array([0, 1, 2])
    elif len(xyz) == 2:
        eqs = plane_from_2points(xyz[:, :3])[None]
        vertices = np.array([0, 1])
    else:
        print(xyz)
        raise RuntimeError('Molecule with less than two atoms.')
    
    if flat:
        planar_seg_eqs, planar_seg_inds = find_planar_segments(xyz, eqs, dist_tol=flat_dist_tol, num_atoms=flat_num_atoms)
        for eq in planar_seg_eqs:
            n_vecs.append(eq[:3])
            sources.append("Planar segment")
        eqs = np.delete(eqs, planar_seg_inds, axis=0)

    if plane_bias:
        plane_elems = get_plane_elements(xyz, eqs, dist_tol=elem_dist_tol)
        for eq, elems in zip(eqs, plane_elems):
            for e, p in plane_bias.items():
                if e in elems and (np.random.rand() <= p):
                    n_vecs.append(eq[:3])
                    sources.append(f"Plane with element {e}")
                    break

    if random_bias:
        elems = set(xyz[vertices, -1].astype(int))
        for e in random_bias:
            if e not in elems:
                continue
            count = random_bias[e]
            while count > 0:
                if count < 1 and np.random.rand() > count:
                    break
                while True:
                    n = random_unit_vector()
                    _, phi, theta = cart_to_sph(n)
                    new_xyz = xyz.copy()
                    new_xyz[:, :3] = np.dot(new_xyz[:, :3], zyz_rotation(-phi, -theta, 0).T)
                    eq = np.array([0, 0, 1, -new_xyz[:, 2].max()])
                    plane_elems = get_plane_elements(new_xyz, [eq], dist_tol=0.7)
                    if e in plane_elems[0]:
                        break
                if len(n_vecs) > 0:
                    n_vecs_np = np.stack(n_vecs, axis=0)
                    angles = np.arccos(np.dot(n_vecs_np, n) / np.linalg.norm(n_vecs_np, axis=1)) / np.pi * 180
                    if all(angles > angle_tolerance):
                        n_vecs.append(n)
                        sources.append(f"Random rotation for element {e}")
                        count -= 1
                else:
                    n_vecs.append(n)
                    sources.append(f"Random rotation for element {e}")
                    count -= 1

    results = []
    for vec, source in zip(n_vecs, sources):
        _, phi, theta = cart_to_sph(vec)
        psi = 0  # As used in zyz_rotation(-phi, -theta, 0)
        R = zyz_rotation(-phi, -theta, psi)
        angles = (-np.degrees(phi), -np.degrees(theta), np.degrees(psi))  # Convert to degrees
        results.append((R, angles, source))

    return results

def process_fchk_to_rotations(fchk_path, **kwargs):
    """
    Process a Gaussian .fchk file and compute rotation matrices and angles.
    
    Args:
        fchk_path (str): Path to the .fchk file.
        **kwargs: Arguments to pass to choose_rotations_bias.
    
    Returns:
        list: List of (rotation_matrix, angles, source) tuples.
    """
    xyz = parse_fchk(fchk_path)
    rotations = choose_rotations_bias(xyz, **kwargs)
    return rotations

# Example usage
if __name__ == "__main__":
    fchk_file = "/scratch/phys/sin/sethih1/data_files/first_group/7923.fchk"
    #fchk_file = "/scratch/phys/sin/sethih1/data_files/first_group/10038.fchk"
    fchk_file = "/scratch/phys/sin/sethih1/data_files/first_group/7492.fchk"
    try:
        np.random.seed(42)  # For reproducibility
        rotations = process_fchk_to_rotations(
            fchk_file,
            flat=True,
            plane_bias={'O': 1, 'C': 0.8, 'H': 0.5, 'N': 0.5},
            random_bias={'O': 0.5, 'C': 0.4, 'H': 2, 'N': 0.5},
            angle_tolerance=5,
            elem_dist_tol=0.7,
            flat_dist_tol=0.1,
            flat_num_atoms=6
        )
        print(f"Number of rotations found: {len(rotations)}")
        for i, (rot, angles, source) in enumerate(rotations, 1):
            phi, theta, psi = angles
            print(f"\nRotation {i} (Source: {source}):")
            print(f"Angles (phi, theta, psi) in degrees: ({phi:.6f}, {theta:.6f}, {psi:.6f})")
            print(f"Rotation matrix:\n{rot}")
            # Verify by applying angles
            xyz = parse_fchk(fchk_file)
            rotated_xyz = apply_zyz_rotation(xyz, phi, theta, psi, deg=True)
            matrix_rotated = xyz[:, :3] @ rot
            if np.allclose(rotated_xyz[:, :3], matrix_rotated):
                print("Verification: Angles produce the same rotation as the matrix.")
            else:
                print("Warning: Angle-based rotation differs from matrix.")
    except Exception as e:
        print(f"Error processing .fchk file: {e}")