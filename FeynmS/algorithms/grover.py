import numpy as np
from typing import List
from ..core.circuit import QuantumCircuit
from ..core.gates import QuantumGate, CustomGate

class GroverSearch:
    """
    A class to represent Grover's search algorithm.

    Attributes:
    num_qubits : int
        The number of qubits in the quantum circuit.
    marked_states : List[str]
        The list of marked states to search for.
    circuit : QuantumCircuit
        The quantum circuit used for the algorithm.
    optimal_iterations : int
        The optimal number of iterations for the algorithm.
    """
    def __init__(self, num_qubits: int, marked_states: List[str]):
        """
        Constructs all the necessary attributes for the GroverSearch object.

        Parameters:
        num_qubits : int
            The number of qubits in the quantum circuit.
        marked_states : List[str]
            The list of marked states to search for.
        """
        self.num_qubits = num_qubits
        self.marked_states = marked_states
        self.circuit = QuantumCircuit(num_qubits, num_qubits)
        self.optimal_iterations = int(np.ceil(np.sqrt(2 ** num_qubits / len(marked_states))))

    def _create_oracle(self) -> QuantumGate:
        """
        Creates the oracle gate for the marked states.

        Returns:
        QuantumGate
            The oracle gate.
        """
        N = 2 ** self.num_qubits
        oracle_matrix = np.eye(N, dtype=complex)
        for state in self.marked_states:
            idx = int(state, 2)
            oracle_matrix[idx, idx] = -1
        return CustomGate.from_matrix(oracle_matrix, "Oracle")

    def _create_diffusion(self) -> QuantumGate:
        """
        Creates the diffusion gate for the algorithm.

        Returns:
        QuantumGate
            The diffusion gate.
        """
        dim = 2 ** self.num_qubits
        diffusion_matrix = 2 * np.ones((dim, dim)) / dim - np.eye(dim)
        return CustomGate.from_matrix(diffusion_matrix, "Diffusion")

    def run(self, measure: bool = True) -> QuantumCircuit:
        """
        Runs Grover's search algorithm.

        Parameters:
        measure : bool, optional
            Whether to measure the qubits at the end (default is True).

        Returns:
        QuantumCircuit
            The quantum circuit after running the algorithm.
        """
        for i in range(self.num_qubits):
            self.circuit.h(i)

        oracle = self._create_oracle()
        diffusion = self._create_diffusion()

        for _ in range(self.optimal_iterations):
            self.circuit.add_gate(oracle, list(range(self.num_qubits)))
            self.circuit.add_gate(diffusion, list(range(self.num_qubits)))

        if measure:
            for i in range(self.num_qubits):
                self.circuit.measure(i, i)

        return self.circuit