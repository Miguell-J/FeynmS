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
        self.unitary = unitary
        self.precision_qubits = precision_qubits
        self.target_qubits = int(np.log2(unitary.matrix.shape[0]))
        self.total_qubits = self.precision_qubits + self.target_qubits
        self.circuit = QuantumCircuit(self.total_qubits, self.precision_qubits)

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
        if initial_state is not None:
            if len(initial_state) != 2 ** self.target_qubits:
                raise ValueError("Initial state must match the target qubits' dimension")
            # Inicializa o registro alvo com o estado fornecido
            init_gate = CustomGate.from_state(initial_state, "InitTarget")
            self.circuit.add_gate(init_gate, list(range(self.precision_qubits, self.total_qubits)))

        # Aplica Hadamard nos qubits de precisão
        for i in range(self.precision_qubits):
            self.circuit.h(i)

        # Aplica as potências controladas de U
        for i in range(self.precision_qubits):
            power_matrix = np.linalg.matrix_power(self.unitary.matrix, 2 ** i)
            power_gate = CustomGate.from_matrix(power_matrix, f"U^{2 ** i}")
            controlled_power = ControlledGate.create_controlled(power_gate)
            control_target_qubits = [i] + list(range(self.precision_qubits, self.total_qubits))
            self.circuit.add_gate(controlled_power, control_target_qubits)

        # Aplica a QFT inversa nos qubits de precisão
        qft_inv = QuantumFourierTransform.create_circuit(self.precision_qubits, inverse=True)
        for op in qft_inv.operations:
            self.circuit.add_gate(op.gate, op.qubits, op.classical_bits)

        # Mede os qubits de precisão
        for i in range(self.precision_qubits):
            self.circuit.measure(i, i)

        return self.circuit