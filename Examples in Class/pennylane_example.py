import pennylane as qml
from pennylane import numpy as np

dev1 = qml.device('default.qubit', wires=1)

"""def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval.PauliZ(0)"""

@qml.qnode(dev1)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval.PauliZ(0)

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [1]]
for x in X:
    print("{0} : {1}".format(x, circuit(x)))