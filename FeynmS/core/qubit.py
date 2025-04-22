import numpy as np
from typing import Union, Optional, Tuple, List
from enum import Enum
import cmath

class BasisState(Enum):
    """Enumeration representing computational and Hadamard basis states."""
    ZERO = '0'
    ONE = '1'
    PLUS = '+'
    MINUS = '-'

class QubitState:
    """Class representing basic qubit states in different bases."""
    
    @staticmethod
    def zero() -> np.ndarray:
        """State |0⟩"""
        return np.array([1, 0], dtype=complex)
    
    @staticmethod
    def one() -> np.ndarray:
        """State |1⟩"""
        return np.array([0, 1], dtype=complex)
    
    @staticmethod
    def plus() -> np.ndarray:
        """State |+⟩ = (|0⟩ + |1⟩)/√2"""
        return np.array([1, 1], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def minus() -> np.ndarray:
        """State |-⟩ = (|0⟩ - |1⟩)/√2"""
        return np.array([1, -1], dtype=complex) / np.sqrt(2)

class MultiQubitState:
    """
    Represents the state of multiple entangled qubits.
    
    Attributes:
        state (np.ndarray): The global state vector of the qubits.
        qubit_names (List[str]): List of qubit names.

    """
    
    def __init__(self, state: np.ndarray, qubit_names: List[str]):
        """
        Initializes a multi-qubit state.
        
        Args:
            state (np.ndarray): The global state vector of the qubits.
            qubit_names (List[str]): List of qubit names.
        """
        self.state = state
        self.qubit_names = qubit_names
        self._validate_state()
    
    def _validate_state(self):
        """Validates the global state vector to ensure normalization."""
        if not isinstance(self.state, np.ndarray):
            raise ValueError("Estado deve ser um numpy array.")
        if not np.isclose(np.sum(np.abs(self.state)**2), 1, rtol=1e-5):
            raise ValueError("Estado do sistema multi-qubit deve ser normalizado.")
    
    def measure(self, qubit_index: int) -> Tuple[int, float]:
        """
        Measures a specific qubit in the current state.
        
        Args:
            qubit_index (int): Index of the qubit to be measured.
            
        Returns:
            Tuple[int, float]: Measurement result (0 or 1) and its probability.
        """
        n_qubits = len(self.qubit_names)
        prob_0 = 0
        # Itera sobre todos os estados possíveis
        for i in range(2**n_qubits):
            # Verifica se o bit no qubit_index é 0
            if not (i & (1 << (n_qubits - 1 - qubit_index))):
                prob_0 += np.abs(self.state[i])**2
        prob_1 = 1 - prob_0
        # Realiza a medição com base nas probabilidades
        result = 1 if np.random.random() < prob_1 else 0
        return result, prob_1 if result == 1 else prob_0
    
    def __str__(self) -> str:
        """Returns a string representation of the multi-qubit state."""
        return f"MultiQubitState(qubits={self.qubit_names}, state={self.state})"

class Qubit:
    """
    Represents an individual qubit in a quantum circuit.
    
    Attributes:
        state (np.ndarray): The qubit state vector.
        name (str): The name of the qubit.
    """
    
    def __init__(self, 
                 state: Optional[Union[np.ndarray, List[complex], BasisState]] = None,
                 name: str = "q"):
        """
        Initializes a qubit with a given state.
        
        Args:
            state (Optional[Union[np.ndarray, List[complex], BasisState]]): 
                The initial state of the qubit, which can be:
                - A state vector (np.ndarray)
                - A list of complex numbers (List[complex])
                - A basis state (BasisState)
                - None (default, initializes in state |0⟩)
            name (str): Identifier name of the qubit.
        """
        self.name = name
        
        if state is None:
            self._state = QubitState.zero()
        elif isinstance(state, BasisState):
            self._state = self._get_basis_state(state)
        elif isinstance(state, (list, np.ndarray)):
            self._state = np.array(state, dtype=complex)
            self._validate_state()
        else:
            raise ValueError("Invalid initial state")
            
        self._normalize()
    
    @property
    def state(self) -> np.ndarray:
        """Returns the qubit state vector."""
        return self._state
    
    @state.setter
    def state(self, new_state: np.ndarray):
        """Sets a new state for the qubit, validating and normalizing it."""
        self._state = np.array(new_state, dtype=complex)
        self._validate_state()
        self._normalize()
    
    def _validate_state(self):
        """Validates the qubit state to ensure it is a normalized 2D vector."""
        if self._state.shape != (2,):
            raise ValueError("The qubit state must be a 2D vector.")
        if not np.isclose(np.sum(np.abs(self._state)**2), 1, rtol=1e-5):
            raise ValueError("The qubit state must be normalized.")
    
    def _normalize(self):
        """Normalizes the state vector to ensure its norm is 1."""
        norm = np.sqrt(np.sum(np.abs(self._state)**2))
        if not np.isclose(norm, 0):
            self._state = self._state / norm
    
    def _get_basis_state(self, basis: BasisState) -> np.ndarray:
        """Returns the corresponding state vector for a computational or Hadamard basis state."""
        basis_states = {
            BasisState.ZERO: QubitState.zero(),
            BasisState.ONE: QubitState.one(),
            BasisState.PLUS: QubitState.plus(),
            BasisState.MINUS: QubitState.minus()
        }
        return basis_states[basis]
    
    def measure(self, basis: BasisState = BasisState.ZERO) -> Tuple[int, float]:
        """
        Measures the qubit in a given basis.
        
        Args:
            basis (BasisState): Measurement basis. Default is the computational basis (|0⟩, |1⟩).
            
        Returns:
            Tuple[int, float]: Measurement result (0 or 1) and its probability.
        """
        if basis in [BasisState.ZERO, BasisState.ONE]:
            prob_1 = np.abs(self._state[1])**2
            result = 1 if np.random.random() < prob_1 else 0
            probability = prob_1 if result == 1 else 1 - prob_1
        else:
            measurement_state = self.change_basis(basis)
            prob_1 = np.abs(measurement_state[1])**2
            result = 1 if np.random.random() < prob_1 else 0
            probability = prob_1 if result == 1 else 1 - prob_1
            
        return result, probability
    
    def change_basis(self, new_basis: BasisState) -> np.ndarray:
        """
        Transforms the qubit state into a new basis.
        
        Args:
            new_basis (BasisState): The new basis for representation.
            
        Returns:
            np.ndarray: The state in the new basis.
        """
        if new_basis in [BasisState.PLUS, BasisState.MINUS]:
            H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            return H @ self._state
        return self._state
    
    def get_bloch_coordinates(self) -> Tuple[float, float, float]:
        """
        Calculates the qubit coordinates on the Bloch sphere.
        
        Returns:
            Tuple[float, float, float]: Coordinates (x, y, z)
        """
        # State parameters |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        theta = 2 * np.arccos(np.abs(self._state[0]))
        phi = cmath.phase(self._state[1]) - cmath.phase(self._state[0])
        
        # Coordenadas da esfera de Bloch
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return x, y, z
    
    def __str__(self) -> str:
        """String representation of the qubit."""
        return f"Qubit(name='{self.name}', state=[{self._state[0]:.3f}, {self._state[1]:.3f}])"
    
    def __repr__(self) -> str:
        """Formal representation of the qubit."""
        return self.__str__()

    def __eq__(self, other: 'Qubit') -> bool:
        """Compares two qubits."""
        if not isinstance(other, Qubit):
            return False
        return np.allclose(self._state, other._state)