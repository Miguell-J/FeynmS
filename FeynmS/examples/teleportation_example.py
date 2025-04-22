from FeynmS.algorithms.teleportation import Teleportation

def main():
    teleport = Teleportation(state_to_teleport=[1, 0])
    circuit = teleport.run()
    print(circuit.draw())

if __name__ == "__main__":
    main()