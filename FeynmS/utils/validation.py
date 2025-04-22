import numpy as np

def check_normalization(state: np.ndarray) -> bool:
    """
    Checks if a quantum state is normalized and returns the calculated norm.

    Parameters:
    state : np.ndarray
        The quantum state to check. Must be a 1D array.

    Returns:
    tuple[bool, float]
        A tuple containing:
            - A boolean indicating whether the state is normalized (True if the norm ≈ 1.0).
            - The calculated norm (sum of the squares of the magnitudes).
    """
    # Achata o array para garantir que seja unidimensional
    state_vector = state.flatten()
    
    norm = np.sum(np.abs(state_vector) ** 2)
    is_normalized = np.isclose(norm, 1.0, atol=1e-10)
    return is_normalized, norm