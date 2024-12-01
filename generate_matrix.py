import numpy as np

def generate_adjacency_matrix(n_computers, connectivity_prob,
                              min_distance, max_distance):
    """Generate a random weighted adjacency matrix for a computer network.
    
    Creates a matrix where:
    - Edge weights represent distances/latencies between computers
    - -1 indicates no connection
    - 0 indicates self-connection
    - Each node has at least one outgoing edge
    """
    while True:
        # Initialize matrix with -1 (no connections)
        matrix = np.full((n_computers, n_computers), -1)

        # Set diagonal to 0 (self-connections)
        np.fill_diagonal(matrix, 0)

        # Generate random connections
        for i in range(n_computers):
            for j in range(n_computers):
                if i != j:
                    # Decide if there's a connection
                    if np.random.random() < connectivity_prob:
                        # Generate random distance between min and max
                        matrix[i,j] = np.random.randint(min_distance, max_distance + 1)

        # Check if graph is minimally connected (at least one outgoing edge per node)
        is_connected = all(any(matrix[i,j] != -1 for j in range(n_computers) if i != j)
                          for i in range(n_computers))
        
        if is_connected:
            return matrix

def save_matrix_to_csv(matrix, filename):
    """Save adjacency matrix to CSV file."""
    np.savetxt(filename, matrix, delimiter=',', fmt='%d')

def load_matrix_from_csv(filename):
    """Load adjacency matrix from CSV file."""
    return np.loadtxt(filename, delimiter=',')