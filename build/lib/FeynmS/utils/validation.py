import numpy as np

def check_normalization(state: np.ndarray) -> bool:
    """
    Checks if a quantum state is normalized.

    Parameters:
    state : np.ndarray
        The quantum state to be checked.

    Returns:
    bool
        True if the state is normalized, False otherwise.
    """
    norm = np.sum(np.abs(state) ** 2)
    return np.isclose(norm, 1.0, atol=1e-10)