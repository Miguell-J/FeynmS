import pytest
from FeynmS.algorithms.grover import GroverSearch

def test_grover_search():
    grover = GroverSearch(num_qubits=2, marked_states=["11"])
    circuit = grover.run()
    assert circuit.num_qubits == 2