# Informações da versão
__version__ = '0.1.0'
__author__ = 'Miguel Araújo Julio'
__email__ = 'Julioaraujo.guel@gmail.com'
__license__ = 'MIT'

# Importações principais
from .core.qubit import Qubit
from .core.gates import QuantumGate, StandardGates, ControlledGate
from .core.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .algorithms.grover import GroverSearch
from .algorithms.qft import QuantumFourierTransform
from .algorithms.phase_estimation import PhaseEstimation
from .algorithms.teleportation import Teleportation

from .visualization.plotting import plot_circuit
from .utils.state_utils import state_to_vector, measure_state