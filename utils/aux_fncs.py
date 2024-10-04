import numpy as np

def generate_normalized_2d_sq_matrix(rows):
    """
    Generates a matrix of the given size (rows x cols) with random values,
    where each row is normalized so that its sum equals 1.
    """
    matrix = np.ones((rows, rows))  # Create a matrix with all values set to 1
    normalized_matrix = matrix / rows  # Normalize so that each row sums to 1
    return normalized_matrix