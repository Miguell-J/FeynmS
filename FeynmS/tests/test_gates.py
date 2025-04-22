import pytest
import numpy as np
from FeynmS.core.gates import QuantumGate, StandardGates

def test_hadamard_gate():
    h_gate = StandardGates.H()
    qubit_state = np.array([1, 0], dtype=complex)
    new_state = h_gate.apply(qubit_state)
    expected_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(new_state, expected_state)