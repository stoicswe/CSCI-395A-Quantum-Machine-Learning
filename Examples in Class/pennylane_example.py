import pennylane as qml
from pennylane import numpy as np

dev1 = qml.device('default.qubit', wires=1)

def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval.PauliZ(0)

@qml.qnode(dev1)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval.PauliZ(0)

while(True):
    print("Input some data for rotation like the following: n1,n2 ")
    nums = input(">")
    nums = nums.split(",")
    n1 = float(nums[0])
    n2 = float(nums[1])
    print("Result of experiment:")
    print(circuit([n1,n2]))
#circuit([0.54, 0.12])