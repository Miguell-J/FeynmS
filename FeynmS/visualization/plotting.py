import matplotlib.pyplot as plt
from ..core.circuit import QuantumCircuit

def plot_circuit(circuit: QuantumCircuit, save_path: str = None):
    """
    Plots the given quantum circuit using matplotlib.

    Parameters:
    circuit : QuantumCircuit
        The quantum circuit to be plotted.
    save_path : str, optional
        If provided, saves the plot to the specified file path instead of displaying it.
    """
    if not circuit.operations or circuit.num_qubits == 0:
        print("Warning: Empty circuit provided. Nothing to plot.")
        return

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(max(10, len(circuit.operations) * 1.5), circuit.num_qubits * 1.5))
    ax.set_title("Quantum Circuit")
    ax.set_xlabel("Operations")
    ax.set_ylabel("Qubits")

    # Desenha linhas horizontais para cada qubit
    for i in range(circuit.num_qubits):
        ax.axhline(y=i, color='black', linewidth=1)

    # Desenha as operações do circuito
    for idx, op in enumerate(circuit.operations):
        if op.gate is not None:
            for qubit in op.qubits:
                ax.text(idx, qubit, op.gate.name, ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Configura os eixos
    ax.set_yticks(range(circuit.num_qubits))
    ax.set_yticklabels([f"q{i}" for i in range(circuit.num_qubits)])
    ax.set_xticks(range(len(circuit.operations)))
    ax.set_xticklabels([f"Op {i}" for i in range(len(circuit.operations))])
    ax.set_ylim(-0.5, circuit.num_qubits - 0.5)
    ax.set_xlim(-0.5, len(circuit.operations) - 0.5)

    # Ajusta o layout e exibe ou salva
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)  # Libera memória fechando a figura