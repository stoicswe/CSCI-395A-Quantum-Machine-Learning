import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
from math import log
from sklearn.datasets import load_iris
import numpy as np

######################################
# INPUT

X = tf.placeholder(tf.float32, [4])
y = tf.placeholder(tf.float32, [3])

######################################
# PARAMETERS
# VGATE
alpha0 = tf.Variable(0.1)
alpha1 = tf.Variable(0.1)
alpha2 = tf.Variable(0.1)
alpha3 = tf.Variable(0.1)
# DGATE
alpha4 = tf.Variable(0.1)
alpha5 = tf.Variable(0.1)
alpha6 = tf.Variable(0.1)
alpha7 = tf.Variable(0.1)
# BS GATE
alpha8 = tf.Variable(0.1)
alpha9 = tf.Variable(0.1)
alpha10 = tf.Variable(0.1)
alpha11 = tf.Variable(0.1)
alpha12 = tf.Variable(0.1)
alpha13 = tf.Variable(0.1)
alpha14 = tf.Variable(0.1)
# SGATE
alpha15 = tf.Variable(0.1)
alpha16 = tf.Variable(0.1)
alpha17 = tf.Variable(0.1)
alpha18 = tf.Variable(0.1)

######################################
# QUANTUM CIRCUIT

eng, q = sf.Engine(4)

with eng:
    Dgate(X[0], 0.) | q[0]
    Dgate(X[1], 0.) | q[1]
    Dgate(X[2], 0.) | q[2]
    Dgate(X[3], 0.) | q[3]

    Vgate(alpha0) | q[0]
    Vgate(alpha1) | q[1]
    Vgate(alpha2) | q[2]
    Vgate(alpha3) | q[3]

    Dgate(alpha4) | q[0]
    Dgate(alpha5) | q[1]
    Dgate(alpha6) | q[2]
    Dgate(alpha7) | q[3]

    BSgate(phi=alpha8) | (q[0], q[1])
    BSgate() | (q[0], q[1])

    BSgate(phi=alpha9) | (q[0], q[2])
    BSgate() | (q[0], q[1])
    
    BSgate(phi=alpha10) | (q[0], q[3])
    BSgate() | (q[0], q[1])
    
    BSgate(phi=alpha12) | (q[1], q[2])
    BSgate() | (q[0], q[1])

    BSgate(phi=alpha13) | (q[1], q[3])
    BSgate() | (q[0], q[1])

    BSgate(phi=alpha14) | (q[2], q[3])
    BSgate() | (q[0], q[1])

    Sgate(alpha15) | q[0]
    Sgate(alpha16) | q[1]
    Sgate(alpha17) | q[2]
    Sgate(alpha18) | q[3]

state = eng.run('tf', cutoff_dim=10, eval=False)
p0 = state.fock_prob([0,0,0,2])
p1 = state.fock_prob([0,0,2,0])
p2 = state.fock_prob([0,2,0,0])
p3 = state.fock_prob([2,0,0,0])
normalization = p0 + p1 + p2 + p3 + 1e-10
circuit_output = [p0 / normalization, p1 / normalization, p2 / normalization]
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=circuit_output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

# Beign the training process
data = load_iris()
xs = data.data[:, :4]
#print(xs)
ys = []
y_origin = data.target
for yi in y_origin:
    if yi == 0:
        ys.append([1.,0.,0.])
    if yi == 1:
        ys.append([0.,1.,0.])
    if yi == 2:
        ys.append([0.,0.,1.])
#ys = np.array(ys)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 500
for s in range(steps):
    for i in range(len(xs)):
        #print("X")
        #print(xs[i])
        #print("Y")
        #print(ys[i])
        sess.run(minimize_op, feed_dict={X:xs[i], y:ys[i]})
    if (s % 10 == 0):
        print("{0}% | Loss: {1}".format((s/steps)*100, sess.run(loss, feed_dict={X:xs[i], y:ys[i]})))

print("Done")