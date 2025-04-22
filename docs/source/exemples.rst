.. _examples:

Examples
========

This section provides practical examples of using FeynmS to implement quantum computing concepts.

Grover's Search Algorithm
-------------------------

Search for a specific state using Grover's algorithm:

.. code-block:: python

   from FeynmS.algorithms.grover import GroverSearch

   # Define a 3-qubit circuit searching for state '101'
   grover = GroverSearch(num_qubits=3, marked_states=['101'])

   # Run the algorithm
   circuit = grover.run()
   print(circuit)

Quantum Teleportation
---------------------

Teleport a quantum state between qubits:

.. code-block:: python

   import numpy as np
   from FeynmS.algorithms.teleportation import Teleportation

   # Define the state to teleport (e.g., (|0⟩ + |1⟩)/√2)
   state_to_teleport = [1/np.sqrt(2), 1/np.sqrt(2)]

   # Create and run the teleportation protocol
   teleportation = Teleportation(state_to_teleport)
   circuit = teleportation.run()
   print(circuit)
