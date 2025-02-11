import matplotlib.pyplot as plt
from ..core.circuit import QuantumCircuit

def plot_circuit(circuit: QuantumCircuit):
    """
    Plots the given quantum circuit.

    Parameters:
    circuit : QuantumCircuit
        The quantum circuit to be plotted.
    """
    try:
        from qiskit.visualization import plot_circuit_layout
        layout = plot_circuit_layout(circuit)
        plt.show()
    except ImportError:
        print("Error: Required packages are not installed.")
        raise