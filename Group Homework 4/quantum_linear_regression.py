import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

X = tf.placeholder(tf.float32, [1])
Y = tf.placeholder(tf.float32, [1])

alpha0 = tf.Variable(0.1)
alpha1 = tf.Variable(0.1)
alpha2 = tf.Variable(0.1)

alpha3 = tf.Variable(0.1)
alpha4 = tf.Variable(0.1)
alpha5 = tf.Variable(0.1)

alpha6 = tf.Variable(0.1)
alpha7 = tf.Variable(0.1)
alpha8 = tf.Variable(0.1)

eng, q = sf.Engine(1)

with eng:
    Dgate(X[0], 0.) | q[0]

    Dgate(alpha1)   | q[0]
    Sgate(alpha2)   | q[0]
    Vgate(alpha0)   | q[0]

    Dgate(alpha4)   | q[0]
    Sgate(alpha5)   | q[0]
    Vgate(alpha3)   | q[0]

    Dgate(alpha7)   | q[0]
    Sgate(alpha8)   | q[0]
    Vgate(alpha6)   | q[0]

state = eng.run('tf', cutoff_dim=10, eval=False)
p0 = state.fock_prob([2])
circuit_output = [p0*10]
squared_delta = tf.square(circuit_output[0]-Y)
cost = tf.reduce_sum(squared_delta)
optimize = tf.train.GradientDescentOptimizer(0.1)
train = optimize.minimize(cost)

train_X = [0.1, 0.4, 0.6, 0.9]
train_Y = [0.2, 0.5, 0.7, 1.0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 1500
for s in range(steps):
    for i in range(len(train_X)):
        sess.run(train, {X: [train_X[i]], Y: [train_Y[i]]})
    if (s % 10 == 0):
        print("{0}% | Cost: {1}".format((s/steps)*100, sess.run(cost, feed_dict={X:[train_X[i]], Y: [train_Y[i]]})))

for i in range(len(train_X)):
    print("Input: {0} Pred: {1} Exact: {2}".format(train_X[i], sess.run(circuit_output, {X: [train_X[i]], Y: [train_Y[i]]}), train_Y[i]))