# Getting Started with FeynmS

## Installation

FeynmS requires Python 3.8 or higher and depends on NumPy and Matplotlib. Install it via pip:

```bash
pip install FeynmS
```

## Quick Exemple

Here's a simple example to create and run a quantum circuit:

```python
from FeynmS.core.circuit import QuantumCircuit

# Create a circuit with 2 qubits and 2 classical bits
circuit = QuantumCircuit(2, 2)

# Add gates: Hadamard on qubit 0, CNOT between qubits 0 and 1
circuit.h(0)
circuit.cx(0, 1)

# Measure both qubits
circuit.measure(0, 0)
circuit.measure(1, 1)

# Execute the circuit (1024 shots)
results = circuit.execute(shots=1024)
print(results)
```

For more exemples, see the section.
