import numpy as np
from typing import List, Optional, Union, Tuple
from .qubit import Qubit, MultiQubitState

class QuantumGate:
    """
    A class to represent a quantum gate.

    Attributes:
    matrix : np.ndarray
        The matrix representation of the quantum gate.
    name : str
        The name of the quantum gate.
    num_qubits : int
        The number of qubits the gate acts on.
    """
    def __init__(self, matrix: np.ndarray, name: str, num_qubits: int = 1):
        """
        Constructs all the necessary attributes for the QuantumGate object.

        Parameters:
        matrix : np.ndarray
            The matrix representation of the quantum gate.
        name : str
            The name of the quantum gate.
        num_qubits : int, optional
            The number of qubits the gate acts on (default is 1).
        """
        self.matrix = np.array(matrix, dtype=complex)
        self.name = name
        self.num_qubits = num_qubits
        self._validate_matrix()

    def _validate_matrix(self):
        dim = 2 ** self.num_qubits
        if self.matrix.shape != (dim, dim):
            raise ValueError(f"Matrix must have dimension {dim}x{dim}")
        if not np.allclose(self.matrix @ self.matrix.conj().T, np.eye(dim)):
            raise ValueError("Matrix must be unitary")

    def apply(self, qubits: List[Qubit]) -> 'MultiQubitState':
        """
        Applies the quantum gate to a qubit or a list of qubits.

        Parameters:
        qubit : Union[Qubit, List[Qubit]]
            The qubit or list of qubits to apply the gate to.

        Returns:
        Union[Qubit, MultiQubitState]
            The resulting qubit or multi-qubit state after applying the gate.
        """
        if len(qubits) != self.num_qubits:
            raise ValueError(f"Gate {self.name} requires {self.num_qubits} qubits")
        
        # Constrói o estado global com produto tensorial
        global_state = qubits[0].state
        for q in qubits[1:]:
            global_state = np.kron(global_state, q.state)
        
        # Aplica a matriz da porta ao estado global
        new_state = np.dot(self.matrix, global_state)
        
        # Normaliza o estado resultante
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        
        # Retorna um MultiQubitState com o novo estado
        return MultiQubitState(new_state, [q.name for q in qubits])

    def __str__(self):
        """
        Returns a string representation of the quantum gate.

        Returns:
        str
            The name of the quantum gate.
        """
        return f"{self.name} Gate"

class StandardGates:
    """
    A class to represent standard quantum gates.
    """
    @staticmethod
    def I() -> QuantumGate:
        return QuantumGate(np.array([[1, 0], [0, 1]], dtype=complex), 'I')

    @staticmethod
    def X() -> QuantumGate:
        return QuantumGate(np.array([[0, 1], [1, 0]], dtype=complex), 'X')

    @staticmethod
    def Y() -> QuantumGate:
        return QuantumGate(np.array([[0, -1j], [1j, 0]], dtype=complex), 'Y')

    @staticmethod
    def Z() -> QuantumGate:
        return QuantumGate(np.array([[1, 0], [0, -1]], dtype=complex), 'Z')

    @staticmethod
    def H() -> QuantumGate:
        return QuantumGate(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), 'H')

    @staticmethod
    def S() -> QuantumGate:
        return QuantumGate(np.array([[1, 0], [0, 1j]], dtype=complex), 'S')

    @staticmethod
    def T() -> QuantumGate:
        return QuantumGate(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex), 'T')

class ControlledGate:
    """
    A class to represent controlled quantum gates.
    """
    @staticmethod
    def create_controlled(gate: QuantumGate) -> QuantumGate:
        """
        Creates a controlled version of a given quantum gate.

        Parameters:
        gate : QuantumGate
            The quantum gate to be controlled.

        Returns:
        QuantumGate
            The controlled quantum gate.
        """
        dim = len(gate.matrix)
        controlled_matrix = np.eye(2 * dim, dtype=complex)
        controlled_matrix[dim:, dim:] = gate.matrix
        return QuantumGate(controlled_matrix, f"C-{gate.name}", gate.num_qubits + 1)

    @staticmethod
    def CNOT() -> QuantumGate:
        return QuantumGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex), "CNOT", 2)

    @staticmethod
    def SWAP() -> QuantumGate:
        return QuantumGate(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex), "SWAP", 2)

class CustomGate(QuantumGate):
    """
    A class to represent a custom quantum gate created from a matrix.

    Attributes:
        matrix : np.ndarray
            The matrix representation of the custom gate.
        name : str
            The name of the custom gate.
        num_qubits : int
            The number of qubits the gate acts on.
    """
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, name: str) -> 'CustomGate':
        """
        Creates an instance of CustomGate from a provided matrix.

        Parameters:
            matrix : np.ndarray
                The matrix representation of the gate.
            name : str
                The name of the gate.

        Returns:
            CustomGate
                An instance of CustomGate.

        Raises:
            ValueError
                If the matrix dimensions are not 2^n x 2^n for some integer n.
        """
        num_qubits = int(np.log2(matrix.shape[0]))
        if 2 ** num_qubits != matrix.shape[0]:
            raise ValueError("Matrix dimensions must be 2^n x 2^n")
        return cls(matrix, name, num_qubits)
