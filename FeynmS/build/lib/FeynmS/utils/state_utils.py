import numpy as np
from typing import List, Dict

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
        raise ValueError("O estado deve ser um numpy array.")
    return state.flatten().tolist()

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
        raise ValueError("O estado deve ser um numpy array.")

    probabilities = np.abs(state.flatten()) ** 2
    results = np.random.choice(range(len(probabilities)), size=shots, p=probabilities)

    counts = {}
    for result in results:
        key = bin(result)[2:].zfill(int(np.log2(len(probabilities))))
        counts[key] = counts.get(key, 0) + 1

    return counts