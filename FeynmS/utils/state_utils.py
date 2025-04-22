import numpy as np
from typing import List, Dict
from .validation import check_normalization

def state_to_vector(state: np.ndarray) -> List[complex]:
    """
    Converts a quantum state represented as a numpy array to a list of complex numbers.

    Parameters:
    state : np.ndarray
        The quantum state to be converted.

    Returns:
    List[complex]
        The quantum state as a list of complex numbers.
    """
    if not isinstance(state, np.ndarray):
        raise ValueError("State must be a numpy array.")
    if state.ndim > 2 or (state.ndim == 2 and min(state.shape) != 1):
        raise ValueError("State must be a vector (unidimensional array or 2D with a dimensão equal to 1).")
    
    # Achata o array para garantir que seja unidimensional
    state_vector = state.flatten()
    
    # Verifica se o tamanho é uma potência de 2 (opcional, apenas um aviso)
    n = len(state_vector)
    if n & (n - 1) != 0:
        print("Alert: The length of the vector is not a power of 2, what is incommon for quantum states.")
    
    return state_vector.tolist()

def measure_state(state: np.ndarray, shots: int = 1024) -> Dict[str, int]:
    """
    Measures a quantum state multiple times and returns the counts of each outcome.

    Parameters:
    state : np.ndarray
        The quantum state to be measured.
    shots : int, optional
        The number of measurement shots (default is 1024).

    Returns:
    Dict[str, int]
        A dictionary with the measurement outcomes as keys and their counts as values.
    """
    if not isinstance(state, np.ndarray):
        raise ValueError("tate must be a numpy array.")

    state_vector = state.flatten()
    
    # Verifica se o estado está normalizado
    if not check_normalization(state_vector):
        raise ValueError("State must be normalized (sum of the squared magnitudes must be 1).")
    
    probabilities = np.abs(state_vector) ** 2
    results = np.random.choice(range(len(probabilities)), size=shots, p=probabilities)

    counts = {}
    num_qubits = int(np.log2(len(probabilities)))
    for result in results:
        key = bin(result)[2:].zfill(num_qubits)
        counts[key] = counts.get(key, 0) + 1

    return counts