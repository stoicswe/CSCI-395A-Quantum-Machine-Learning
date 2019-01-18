import numpy as np
import math
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer # import the prerequisits from qiskit
import matplotlib.pyplot as plt

iterations = 1000
q1 = []
q2 = []
q3 = []
q4 = []
q5 = []
q6 = []
q7 = []
q8 = []
theta_list = [1.2491, 1.1072, 0.9911, 0.8861, 0.7853, 0.6847, 0.5797, 0.4637, 0.3220]
pattern = '10101' # 10101
for i in range(len(theta_list)):
    theta = theta_list[i]
    print("Corruption Theta: {0}".format(theta))
    print("----------------------------------")
    pat = 0
    for r in range(iterations):
        q = QuantumRegister(8)
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
        # add a corruption rate
        enp.u3(theta,0,0, q[5])
        enp.u3(theta,0,0, q[6])
        enp.u3(theta,0,0, q[7])
        enp.cx(q[5],q[4])
        enp.cy(q[6],q[4])
        enp.cz(q[7],q[4])
        # measure the gates, but only the ones that need to be measured 
        enp.measure(q[0],c[0])
        enp.measure(q[1],c[1])
        enp.measure(q[2],c[2])
        enp.measure(q[3],c[3])
        enp.measure(q[4],c[4])
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
    if i == 5:
        q6.append(pat/iterations)
    if i == 6:
        q7.append(pat/iterations)
    if i == 7:
        q8.append(pat/iterations)
print("----------------------------------")
print(q1)
print(q2)
print(q3)
print(q4)
print(q5)
print(q6)
print(q7)
print(q8)
print("----------------------------------")
"""plt.plot(theta_list, q1)
plt.show()

plt.plot(x_facs, q2)
plt.show()

plt.plot(x_facs, q3)
plt.show()

plt.plot(x_facs, q4)
plt.show()

plt.plot(x_facs, q5)
plt.show()"""