import numpy as np
from ..core.circuit import QuantumCircuit
from ..core.gates import QuantumGate, CustomGate, ControlledGate

class QuantumFourierTransform:
    """
    A class to represent the Quantum Fourier Transform (QFT) algorithm.
    """
    @staticmethod
    def create_circuit(num_qubits: int, inverse: bool = False) -> QuantumCircuit:
        """
        Creates a quantum circuit for the Quantum Fourier Transform.

        Parameters:
        num_qubits : int
            The number of qubits in the circuit.
        inverse : bool, optional
            Whether to create the inverse QFT circuit (default is False).

        Returns:
        QuantumCircuit
            The quantum circuit for the QFT.
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be positive")

        circuit = QuantumCircuit(num_qubits)

        def add_qft_gates(start: int, end: int, step: int):
            """
            Adds the QFT gates to the circuit.

            Parameters:
            start : int
                The starting qubit index.
            end : int
                The ending qubit index.
            step : int
                The step size for the qubit indices.
            """
            for i in range(start, end, step):
                circuit.h(i)
                for j in range(i + step, end, step):
                    phase = np.pi / float(2 ** (j - i))
                    rotation_matrix = np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
                    rotation_gate = CustomGate.from_matrix(rotation_matrix, f"R{phase:.2f}")
                    controlled_rotation = ControlledGate.create_controlled(rotation_gate)
                    circuit.add_gate(controlled_rotation, [i, j])

        if not inverse:
            add_qft_gates(0, num_qubits, 1)
            for i in range(num_qubits // 2):
                circuit.add_gate(ControlledGate.SWAP(), [i, num_qubits - 1 - i])
        else:
            for i in range((num_qubits - 1) // 2, -1, -1):
                circuit.add_gate(ControlledGate.SWAP(), [i, num_qubits - 1 - i])
            add_qft_gates(num_qubits - 1, -1, -1)

        return circuit