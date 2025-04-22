from FeynmS.algorithms.grover import GroverSearch

def main():
    grover = GroverSearch(num_qubits=3, marked_states=["101"])
    circuit = grover.run()
    print(circuit.draw())

if __name__ == "__main__":
    main()