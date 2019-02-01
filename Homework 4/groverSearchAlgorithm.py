import numpy as np
import math
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer # import the prerequisits from qiskit

def reverseString(string):
    rs = ""
    for c in string:
        rs = c + rs
    return rs

q = QuantumRegister(3)
c = ClassicalRegister(3)
init = QuantumCircuit(q,c)
oracle = QuantumCircuit(q, c)
us = QuantumCircuit(q,c)
mes = QuantumCircuit(q,c)
bit_combos = ['000', '100', '110', '011', '101', '010', '001', '111']
# BUILD THE THINGS
circuits = []
circuit_collection = []
# initialize the initial setup
init.h(q[0])
init.h(q[1])
init.h(q[2])
# initialize the US
us.h(q[0])
us.h(q[1])
us.h(q[2])
us.x(q[0])
us.x(q[1])
us.x(q[2])
us.h(q[2])
oracle.barrier()
us.ccx(q[0], q[1], q[2])
us.h(q[2])
us.x(q[0])
us.x(q[1])
us.x(q[2])
us.h(q[0])
us.h(q[1])
us.h(q[2])
# initialize the measures
mes.measure(q,c)
for n in range(3):
    for i in range(8):
        curr_bits = bit_combos[i]
        for j in range(len(curr_bits)):
            if curr_bits[j] == '0':
                oracle.y(q[j])
        oracle.h(q[2])
        oracle.barrier()
        oracle.ccx(q[0],q[1],q[2])
        oracle.h(q[2])
        for j in range(len(curr_bits)):
            if curr_bits[j] == '0':
                oracle.y(q[j])
        if n == 0:
            circuits.append(init + oracle + us + mes)
        if n == 1:
            circuits.append(init + oracle + us + oracle + us + mes)
        if n == 2:
            circuits.append(init + oracle + us + oracle + us + oracle + us + mes)
        oracle = QuantumCircuit(q,c)
    circuit_collection.append(circuits)
    circuits = []
# after building, run the circuits
#for c in circuits:
#    print(c.draw())
#print(circuit_collection)
for n in range(3):
    print("===========================")
    print("Running with {0} oracles...".format(n+1))
    shots = 1024
    job = execute(circuit_collection[n], backend = BasicAer.get_backend('qasm_simulator'), shots=shots, seed=8)
    result = job.result()
    for i in range(len(circuit_collection[n])):
        data = result.get_counts(circuit_collection[n][i])
        v, k = max((v, k) for k, v in data.items())
        #print(data)
        print("Recieved {0} {1}: Entered {2} Equal: {3}".format(reverseString(k), v/shots, bit_combos[i], reverseString(k) == bit_combos[i]))