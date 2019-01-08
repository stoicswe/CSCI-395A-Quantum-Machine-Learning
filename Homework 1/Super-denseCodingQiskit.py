# Super-dense coding quantum algorithm
import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer # import the prerequisits from qiskit
#from qiskit.tools.visualization import plot_state_city

q = QuantumRegister(4)
c = ClassicalRegister(2)
sdc = QuantumCircuit(q, c)
# input
#sdc.x(q[0])
#sdc.x(q[1])
# =======================
sdc.h(q[2])
sdc.cx(q[2], q[3])
sdc.cu3(0, math.pi, 0, q[0], q[2])
sdc.cx(q[1], q[2])
sdc.cx(q[2], q[3])
sdc.h(q[2])
sdc.measure(q[2], c[0])
sdc.measure(q[3], c[1])
print(sdc.draw())

job = execute(sdc, backend = BasicAer.get_backend('qasm_simulator'), shots=1024)
result = job.result()
print("Result of the computation")
print(result.get_counts(sdc))
#outputstate = result.get_statevector(sdc, decimals=3)
#plot_state_city(outputstate)