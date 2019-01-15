# Super-dense coding quantum algorithm
import numpy as np
import math
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer # import the prerequisits from qiskit
import matplotlib.pyplot as plt

x_fac = 0.0
iterations = 1000
q1 = []
q2 = []
q3 = []
q4 = []
q5 = []
x_facs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pattern = '10101' # 10101
for x_fac in x_facs:
    print("Corruption Rate: {0}".format(x_fac))
    print("----------------------------------")
    for i in range(5):
        pat = 0
        for r in range(iterations):
            q = QuantumRegister(5)
            c = ClassicalRegister(5)
            enp = QuantumCircuit(q, c)
            enp.h(q[0])
            enp.h(q[1])
            enp.h(q[2])
            enp.h(q[3])
            enp.cx(q[2], q[4])
            enp.cx(q[0], q[4])
            enp.h(q[0])
            enp.h(q[1])
            enp.h(q[2])
            enp.h(q[3])
            enp.h(q[4])
            if (np.random.uniform(0,1) < x_fac):
                enp.x(q[i])
            enp.measure(q,c)
            #print(enp.draw())
            job = execute(enp, backend = BasicAer.get_backend('qasm_simulator'), shots=1)
            result = job.result()
            rc = result.get_counts(enp)
            k = list(rc.keys())
            if (k[0] == pattern):
                pat += 1
        print("{0} Accuracy: {1}".format(i, pat/iterations))
        if i == 0:
            q1.append(pat/iterations)
        if i == 1:
            q2.append(pat/iterations)
        if i == 2:
            q3.append(pat/iterations)
        if i == 3:
            q4.append(pat/iterations)
        if i == 4:
            q5.append(pat/iterations)
    print("----------------------------------")
    print(q1)
    print(q2)
    print(q3)
    print(q4)
    print(q5)
    print("----------------------------------")
plt.plot(x_facs, q1)
plt.show()

plt.plot(x_facs, q2)
plt.show()

plt.plot(x_facs, q3)
plt.show()

plt.plot(x_facs, q4)
plt.show()

plt.plot(x_facs, q5)
plt.show()