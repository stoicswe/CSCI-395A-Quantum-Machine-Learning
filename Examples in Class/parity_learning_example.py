import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, [1])
Y = tf.placeholder(tf.float32, [1])

alpha0 = tf.Variable(0.1)
alpha1 = tf.Variable(0.1)
alpha2 = tf.Variable(0.1)

eng, q = sf.Engine(1)

with eng:
    Dgate(X[0], 0.) | q[0]
    Vgate(alpha0)   | q[0]
    Dgate(alpha1)   | q[0]
    Sgate(alpha2)   | q[0]

state = eng.run('tf', cutoff_dim=10, eval=False)
p0 = state.fock_prob([2])
circuit_output = [p0]
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=circuit_output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

xs = [[0], [1]]
ys = [[1], [0]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 500
for s in range(steps):
    for i in range(len(xs)):
        sess.run(minimize_op, feed_dict={X:xs[i], Y:ys[i]})
    if (s % 10 == 0):
        print("{0}% | Loss: {1}".format((s/steps)*100, sess.run(loss, feed_dict={X:xs[i], Y:ys[i]})))