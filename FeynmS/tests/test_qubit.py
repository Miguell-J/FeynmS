import pytest
import numpy as np
from FeynmS.core.qubit import Qubit, BasisState

def test_qubit_initialization():
    qubit = Qubit()
    assert np.allclose(qubit.state, np.array([1, 0], dtype=complex))

def test_qubit_measurement():
    qubit = Qubit(BasisState.PLUS)
    result, probability = qubit.measure(BasisState.PLUS)
    assert result in [0, 1]
    assert 0 <= probability <= 1