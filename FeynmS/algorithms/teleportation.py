from typing import List, Optional
from ..core.circuit import QuantumCircuit
from ..core.gates import QuantumGate, StandardGates, ControlledGate, CustomGate

class Teleportation:
    """
    A class to represent the quantum teleportation algorithm.

    Attributes:
    circuit : QuantumCircuit
        The quantum circuit used for the teleportation.
    state_to_teleport : Optional[List[complex]]
        The state to be teleported.
    """
    def __init__(self, state_to_teleport: Optional[List[complex]] = None):
        """
        Constructs all the necessary attributes for the Teleportation object.

        Parameters:
        state_to_teleport : Optional[List[complex]], optional
            The state to be teleported (default is None).
        """
        self.circuit = QuantumCircuit(3, 2)
        self.state_to_teleport = state_to_teleport
        if state_to_teleport is not None:
            if len(state_to_teleport) != 2:
                raise ValueError("State to teleport must be a 2-dimensional complex vector")

    def run(self) -> QuantumCircuit:
        """
        Runs the quantum teleportation algorithm.

        Returns:
        QuantumCircuit
            The quantum circuit after running the teleportation algorithm.
        """
        if self.state_to_teleport is not None:
            # Inicializa o qubit 0 com o estado a ser teletransportado
            init_gate = CustomGate.from_state(self.state_to_teleport, "InitTeleport")
            self.circuit.add_gate(init_gate, [0])

        # Cria o par de Bell entre qubits 1 e 2
        self.circuit.h(1)
        self.circuit.add_gate(ControlledGate.CNOT(), [1, 2])

        # Entrelaça o qubit 0 com o par de Bell
        self.circuit.add_gate(ControlledGate.CNOT(), [0, 1])
        self.circuit.h(0)

        # Mede os qubits 0 e 1
        self.circuit.measure(0, 0)
        self.circuit.measure(1, 1)

        # As correções X e Z devem ser aplicadas condicionalmente durante a execução,
        # com base nos bits clássicos medidos (fora deste método).

        return self.circuit