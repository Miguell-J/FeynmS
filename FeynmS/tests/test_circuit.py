import pytest
from FeynmS.core.circuit import QuantumCircuit
from FeynmS.core.gates import StandardGates

def test_circuit_initialization():
    circuit = QuantumCircuit(2)
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2

def test_circuit_h_gate():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    assert len(circuit.operations) == 1
    assert circuit.operations[0].gate.name == "H"