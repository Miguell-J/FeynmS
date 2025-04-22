# FeynmS - Quantum Laboratory

FeynmS is a Python library for simulation and experimentation with quantum computing. It provides tools for creating and manipulating quantum circuits, implementing quantum algorithms, and visualizing results.

## Installation

You can install FeynmS with pip:

```bash
pip install FeynmS
```

## Features

- **Quantum Circuits**: Create and manipulate quantum circuits with ease.
- **Quantum Gates**: Utilize a variety of standard and custom quantum gates.
- **Quantum Algorithms**: Implement algorithms such as Grover's search and Quantum Fourier Transform.
- **Quantum Teleportation**: Simulate quantum teleportation protocols.
- **Visualization**: Plot and visualize quantum circuits.

## Usage

### Creating a Quantum Circuit

```python
from FeynmS.core.circuit import QuantumCircuit

# Create a quantum circuit with 3 qubits and 2 classical bits
circuit = QuantumCircuit(3, 2)

# Add gates to the circuit
circuit.h(0)
circuit.cx(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)

# Execute the circuit
results = circuit.execute(shots=1024)
print(results)
```

### Implementing Grover's Algorithm

```python
from FeynmS.algorithms.grover import GroverSearch

# Define the number of qubits and the marked states
num_qubits = 3
marked_states = ['101']

# Create a GroverSearch object
grover = GroverSearch(num_qubits, marked_states)

# Run the algorithm
circuit = grover.run()
print(circuit)
```

### Quantum Teleportation

```python
from FeynmS.algorithms.teleportation import Teleportation

# Define the state to be teleported
state_to_teleport = [1/np.sqrt(2), 1/np.sqrt(2)]

# Create a Teleportation object
teleportation = Teleportation(state_to_teleport)

# Run the teleportation protocol
circuit = teleportation.run()
print(circuit)
```

## Documentation

Explore the full documentation at [https://feynms.readthedocs.io/](https://feynms.readthedocs.io/).

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](#) file for details.
