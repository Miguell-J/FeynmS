import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from .qubit import Qubit, MultiQubitState
from .gates import QuantumGate, StandardGates, ControlledGate
import copy

class QuantumRegister:
    """
    A class to represent a quantum register.

    Attributes:
    size : int
        The number of qubits in the register.
    name : str
        The name of the register.
    qubits : List[Qubit]
        The list of qubits in the register.
    """
    def __init__(self, size: int, name: str = "q"):
        """
        Constructs all the necessary attributes for the QuantumRegister object.

        Parameters:
        size : int
            The number of qubits in the register.
        name : str, optional
            The name of the register (default is "q").
        """
        self.size = size
        self.name = name
        self.qubits = [Qubit(name=f"{name}_{i}") for i in range(size)]

    def __getitem__(self, index: int) -> Qubit:
        """
        Gets the qubit at the specified index.

        Parameters:
        index : int
            The index of the qubit.

        Returns:
        Qubit
            The qubit at the specified index.
        """
        return self.qubits[index]

    def __setitem__(self, index: int, value: Qubit):
        """
        Sets the qubit at the specified index.

        Parameters:
        index : int
            The index of the qubit.
        value : Qubit
            The qubit to set at the specified index.
        """
        self.qubits[index] = value

    def __len__(self) -> int:
        """
        Returns the number of qubits in the register.

        Returns:
        int
            The number of qubits in the register.
        """
        return self.size

class ClassicalRegister:
    """
    A class to represent a classical register.

    Attributes:
    size : int
        The number of bits in the register.
    name : str
        The name of the register.
    bits : List[int]
        The list of bits in the register.
    """
    def __init__(self, size: int, name: str = "c"):
        """
        Constructs all the necessary attributes for the ClassicalRegister object.

        Parameters:
        size : int
            The number of bits in the register.
        name : str, optional
            The name of the register (default is "c").
        """
        self.size = size
        self.name = name
        self.bits = [0] * size

    def __getitem__(self, index: int) -> int:
        """
        Gets the bit at the specified index.

        Parameters:
        index : int
            The index of the bit.

        Returns:
        int
            The bit at the specified index.
        """
        return self.bits[index]

    def __setitem__(self, index: int, value: int):
        """
        Sets the bit at the specified index.

        Parameters:
        index : int
            The index of the bit.
        value : int
            The bit to set at the specified index.
        """
        if value not in [0, 1]:
            raise ValueError("Classical bit must be 0 or 1")
        self.bits[index] = value

    def __len__(self) -> int:
        """
        Returns the number of bits in the register.

        Returns:
        int
            The number of bits in the register.
        """
        return self.size

class Operation:
    """
    A class to represent a quantum operation.

    Attributes:
    gate : QuantumGate
        The quantum gate to be applied.
    qubits : List[int]
        The list of qubits the gate acts on.
    classical_bits : Optional[List[int]]
        The list of classical bits involved in the operation.
    """
    def __init__(self, gate: QuantumGate, qubits: List[int], classical_bits: Optional[List[int]] = None):
        """
        Constructs all the necessary attributes for the Operation object.

        Parameters:
        gate : QuantumGate
            The quantum gate to be applied.
        qubits : List[int]
            The list of qubits the gate acts on.
        classical_bits : Optional[List[int]], optional
            The list of classical bits involved in the operation (default is None).
        """
        self.gate = gate
        self.qubits = qubits
        self.classical_bits = classical_bits or []

class QuantumCircuit:
    """
    A class to represent a quantum circuit.

    Attributes:
    qreg : QuantumRegister
        The quantum register of the circuit.
    creg : ClassicalRegister
        The classical register of the circuit.
    operations : List[Operation]
        The list of operations in the circuit.
    _standard_gates : StandardGates
        The standard quantum gates.
    """
    def __init__(self, quantum_register: Union[QuantumRegister, int], classical_register: Optional[Union[ClassicalRegister, int]] = None):
        """
        Constructs all the necessary attributes for the QuantumCircuit object.

        Parameters:
        quantum_register : Union[QuantumRegister, int]
            The quantum register or its size.
        classical_register : Optional[Union[ClassicalRegister, int]], optional
            The classical register or its size (default is None).
        """
        if isinstance(quantum_register, int):
            self.qreg = QuantumRegister(quantum_register)
        else:
            self.qreg = quantum_register

        if classical_register is None:
            self.creg = ClassicalRegister(self.qreg.size)
        elif isinstance(classical_register, int):
            self.creg = ClassicalRegister(classical_register)
        else:
            self.creg = classical_register

        self.operations: List[Operation] = []
        self._standard_gates = StandardGates()

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits in the quantum register.

        Returns:
        int
            The number of qubits in the quantum register.
        """
        return self.qreg.size

    @property
    def num_clbits(self) -> int:
        """
        Returns the number of bits in the classical register.

        Returns:
        int
            The number of bits in the classical register.
        """
        return self.creg.size

    def add_gate(self, gate: Union[QuantumGate, str], qubits: Union[int, List[int]], classical_bits: Optional[Union[int, List[int]]] = None):
        """
        Adds a gate to the circuit.

        Parameters:
        gate : Union[QuantumGate, str]
            The quantum gate or its name.
        qubits : Union[int, List[int]]
            The qubit or list of qubits the gate acts on.
        classical_bits : Optional[Union[int, List[int]]], optional
            The classical bit or list of bits involved in the operation (default is None).
        """
        if isinstance(qubits, int):
            qubits = [qubits]
        if isinstance(classical_bits, int):
            classical_bits = [classical_bits]

        if isinstance(gate, str):
            gate = getattr(StandardGates, gate.upper())()

        if len(qubits) != gate.num_qubits:
            raise ValueError(f"Gate {gate.name} must have {gate.num_qubits} qubits")

        self.operations.append(Operation(gate, qubits, classical_bits))

    def h(self, qubit: int):
        """
        Adds a Hadamard gate to the circuit.

        Parameters:
        qubit : int
            The qubit the gate acts on.
        """
        self.add_gate(StandardGates.H(), qubit)

    def x(self, qubit: int):
        """
        Adds a Pauli-X gate to the circuit.

        Parameters:
        qubit : int
            The qubit the gate acts on.
        """
        self.add_gate(StandardGates.X(), qubit)

    def cx(self, control: int, target: int):
        """
        Adds a controlled-NOT gate to the circuit.

        Parameters:
        control : int
            The control qubit.
        target : int
            The target qubit.
        """
        self.add_gate(ControlledGate.CNOT(), [control, target])

    def measure(self, qubit: int, cbit: int):
        """
        Adds a measurement operation to the circuit.

        Parameters:
        qubit : int
            The qubit to be measured.
        cbit : int
            The classical bit to store the measurement result.
        """
        measure_gate = QuantumGate(np.eye(2), "Measure", num_qubits=1)
        self.add_gate(measure_gate, qubit, cbit)

    def execute(self, shots: int = 1) -> Dict[str, int]:
        """
        Executes the quantum circuit.

        Parameters:
        shots : int, optional
            The number of times to run the circuit (default is 1).

        Returns:
        Dict[str, int]
            The measurement results.
        """
        results = {}
        for _ in range(shots):
            # Estado inicial |00...0⟩
            global_state = np.zeros(2**self.num_qubits, dtype=complex)
            global_state[0] = 1.0
            measured_bits = [0] * self.creg.size

            for op in self.operations:
                if op.gate is None:  # Medição
                    qubit_idx = op.qubits[0]
                    cbit_idx = op.classical_bits[0]
                    prob_0 = sum(np.abs(global_state[i])**2 
                                 for i in range(2**self.num_qubits) 
                                 if not (i & (1 << (self.num_qubits - 1 - qubit_idx))))
                    prob_1 = 1 - prob_0
                    result = 1 if np.random.random() < prob_1 else 0
                    measured_bits[cbit_idx] = result
                    global_state = self._collapse_state(global_state, qubit_idx, result)
                else:
                    # Aplicação da porta
                    gate_matrix = self._expand_gate(op.gate, op.qubits)
                    global_state = gate_matrix @ global_state
                    global_state /= np.linalg.norm(global_state)

            result_str = ''.join(map(str, measured_bits))
            results[result_str] = results.get(result_str, 0) + 1
        return results
    
    def _expand_gate(self, gate: QuantumGate, qubit_indices: List[int]) -> np.ndarray:
        if gate.num_qubits == 1:
            matrix = gate.matrix
            for i in range(self.num_qubits):
                if i not in qubit_indices:
                    matrix = np.kron(matrix, np.eye(2))
            return matrix
        elif gate.num_qubits == 2:
            # Simplificação: assume qubits consecutivos
            return np.kron(np.eye(2**(self.num_qubits - 2)), gate.matrix)
        else:
            raise NotImplementedError("Gates with more than 2 qubits not yet supported")

    def _collapse_state(self, state: np.ndarray, qubit_idx: int, result: int) -> np.ndarray:
        new_state = state.copy()
        for i in range(len(state)):
            if ((i >> (self.num_qubits - 1 - qubit_idx)) & 1) != result:
                new_state[i] = 0
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        return new_state

    def draw(self) -> str:
        """
        Draws the quantum circuit.

        Returns:
        str
            The string representation of the circuit.
        """
        circuit_str = ["Quantum Circuit:", "-" * 50]
        qubit_lines = {i: [] for i in range(self.num_qubits)}
        for op in self.operations:
            gate = op.gate.name
            for qubit in op.qubits:
                if gate == "H":
                    qubit_lines[qubit].append("H")
                elif gate == "X":
                    qubit_lines[qubit].append("X")
                elif gate == "CNOT":
                    if qubit == op.qubits[0]:
                        qubit_lines[qubit].append("C")
                    elif qubit == op.qubits[1]:
                        qubit_lines[qubit].append("X")
                elif gate == "Y":
                    qubit_lines[qubit].append("Y")
                elif gate == "Z":
                    qubit_lines[qubit].append("Z")
        for i in range(self.num_qubits):
            line = f"q_{i}: " + " ".join(qubit_lines[i])
            circuit_str.append(line)
        circuit_str.append("-" * 50)
        return "\n".join(circuit_str)

    def __str__(self):
        """
        Returns a string representation of the quantum circuit.

        Returns:
        str
            The string representation of the quantum circuit.
        """
        return f"QuantumCircuit({self.num_qubits} qubits, {self.num_clbits} bits)"