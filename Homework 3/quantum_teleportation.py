import numpy as np
import math
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer # import the prerequisits from qiskit

q = QuantumRegister(5)
c = ClassicalRegister(5)
qc = QuantumCircuit(q, c)

#input
qc.x(q[0])
#-------------------------------------------
qc.h(q[1])
qc.cx(q[1],q[2])
qc.cx(q[0],q[1])
qc.h(q[0])
qc.cx(q[0], q[3])
qc.cx(q[1], q[4])
qc.cx(q[3], q[2])
qc.cz(q[4], q[2])
qc.measure(q[0], c[0])
qc.measure(q[2], c[2])
print(qc.draw())
job = execute(qc, backend = BasicAer.get_backend('qasm_simulator'), shots=8192)
result = job.result()
rc = result.get_counts(qc)
print(rc)