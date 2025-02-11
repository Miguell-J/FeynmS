import numpy as np
from typing import List, Optional
from ..core.circuit import QuantumCircuit
from ..core.gates import QuantumGate, CustomGate, ControlledGate
from .qft import QuantumFourierTransform

class PhaseEstimation:
    """
    A class to represent the phase estimation algorithm.

    Attributes:
    unitary : QuantumGate
        The unitary gate whose eigenphase is to be estimated.
    precision_qubits : int
        The number of qubits used for precision in the estimation.
    target_qubits : int
        The number of qubits in the target register.
    total_qubits : int
        The total number of qubits in the circuit.
    circuit : QuantumCircuit
        The quantum circuit used for the algorithm.
    """
    def __init__(self, unitary: QuantumGate, precision_qubits: int):
        """
        Constructs all the necessary attributes for the PhaseEstimation object.

        Parameters:
        unitary : QuantumGate
            The unitary gate whose eigenphase is to be estimated.
        precision_qubits : int
            The number of qubits used for precision in the estimation.
        """
        if precision_qubits < 1:
            raise ValueError("Number of precision qubits must be positive")
        n = int(np.log2(len(unitary.matrix)))
        if 2 ** n != len(unitary.matrix):
            raise ValueError("Unitary matrix must be square with dimensions 2^n x 2^n")

        self.unitary = unitary
        self.precision_qubits = precision_qubits
        self.target_qubits = n
        self.total_qubits = precision_qubits + n
        self.circuit = QuantumCircuit(self.total_qubits)

    def run(self, initial_state: Optional[List[complex]] = None) -> QuantumCircuit:
        """
        Runs the phase estimation algorithm.

        Parameters:
        initial_state : Optional[List[complex]], optional
            The initial state of the target register (default is None).

        Returns:
        QuantumCircuit
            The quantum circuit after running the algorithm.
        """
        for i in range(self.precision_qubits):
            self.circuit.h(i)

        if initial_state is not None:
            if len(initial_state) != 2 ** self.target_qubits:
                raise ValueError("Initial state must have the same size as the target qubits")

        for i in range(self.precision_qubits):
            power_matrix = np.linalg.matrix_power(self.unitary.matrix, 2 ** i)
            power_gate = CustomGate.from_matrix(power_matrix, f"U^{2 ** i}")
            controlled_power = ControlledGate.create_controlled(power_gate)
            control_target_qubits = [i] + list(range(self.precision_qubits, self.total_qubits))
            self.circuit.add_gate(controlled_power, control_target_qubits)

        qft_inv = QuantumFourierTransform.create_circuit(self.precision_qubits, inverse=True)
        for op in qft_inv.operations:
            self.circuit.add_gate(op.gate, op.qubits, op.classical_bits)

        for i in range(self.precision_qubits):
            self.circuit.measure(i, i)

        return self.circuit