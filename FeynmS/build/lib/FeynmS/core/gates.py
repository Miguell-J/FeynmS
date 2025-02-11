import numpy as np
from typing import List, Optional, Union, Tuple
from qubit import Qubit, MultiQubitState

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

    def apply(self, qubit: Union[Qubit, List[Qubit]]) -> Union[Qubit, MultiQubitState]:
        """
        Applies the quantum gate to a qubit or a list of qubits.

        Parameters:
        qubit : Union[Qubit, List[Qubit]]
            The qubit or list of qubits to apply the gate to.

        Returns:
        Union[Qubit, MultiQubitState]
            The resulting qubit or multi-qubit state after applying the gate.
        """
        if isinstance(qubit, Qubit):
            new_state = np.dot(self.matrix, qubit.state)
            return Qubit(new_state, name=qubit.name)
        elif isinstance(qubit, list):
            if self.num_qubits == 2:
                qubit1, qubit2 = qubit
                global_state = np.kron(qubit1.state, qubit2.state)
                new_global_state = np.dot(self.matrix, global_state)
                norm = np.linalg.norm(new_global_state)
                new_global_state = new_global_state / norm
                return MultiQubitState(new_global_state, [qubit1.name, qubit2.name])
            else:
                return [self.apply(q) for q in qubit]
        else:
            raise ValueError("Input must be a Qubit or a list of Qubits")

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