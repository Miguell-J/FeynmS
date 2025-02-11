from typing import List, Optional
from ..core.circuit import QuantumCircuit
from ..core.gates import QuantumGate, StandardGates, ControlledGate

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
            pass  # TODO: Implementar inicialização do estado arbitrário

        self.circuit.h(1)
        self.circuit.add_gate(ControlledGate.CNOT(), [1, 2])

        self.circuit.add_gate(ControlledGate.CNOT(), [0, 1])
        self.circuit.h(0)

        self.circuit.measure(0, 0)
        self.circuit.measure(1, 1)

        x_gate = StandardGates.X()
        controlled_x = ControlledGate.create_controlled(x_gate)
        self.circuit.add_gate(controlled_x, [1, 2])

        z_gate = StandardGates.Z()
        controlled_z = ControlledGate.create_controlled(z_gate)
        self.circuit.add_gate(controlled_z, [0, 2])

        return self.circuit