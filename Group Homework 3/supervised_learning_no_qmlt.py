import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
from math import log

eng, q = sf.Engine(2)

alpha0 = tf.Variable(0.1)
alpha1 = tf.Variable(0.1)
alpha2 = tf.Variable(0.1)
alpha3 = tf.Variable(0.1)
alpha4 = tf.Variable(0.1)
alpha5 = tf.Variable(0.1)
alpha6 = tf.Variable(0.1)
alpha7 = tf.Variable(0.1)
alpha8 = tf.Variable(0.1)

beta0 = tf.Variable(0.1)
beta1 = tf.Variable(0.1)
beta2 = tf.Variable(0.1)
beta3 = tf.Variable(0.1)
beta4 = tf.Variable(0.1)
beta5 = tf.Variable(0.1)
beta6 = tf.Variable(0.1)
beta7 = tf.Variable(0.1)

gamma0 = tf.Variable(0.1)
gamma1 = tf.Variable(0.1)
gamma2 = tf.Variable(0.1)
gamma3 = tf.Variable(0.1)
gamma4 = tf.Variable(0.1)
gamma5 = tf.Variable(0.1)
gamma6 = tf.Variable(0.1)
gamma7 = tf.Variable(0.1)


X = tf.placeholder(tf.float32, [2])
y = tf.placeholder(tf.float32, [2])

with eng:
    Dgate(X[0], 0.) | q[0]
    Dgate(X[1], 0.) | q[1]

    Vgate(alpha0) | q[0]
    Vgate(alpha1) | q[1]
    Dgate(alpha3) | q[0]
    Dgate(alpha4) | q[1]
    BSgate(phi=alpha5) | (q[0], q[1])
    BSgate() | (q[0], q[1])
    Sgate(alpha6) | q[0]
    Sgate(alpha7) | q[1]

    Vgate(beta0) | q[0]
    Vgate(beta1) | q[1]
    Dgate(beta3) | q[0]
    Dgate(beta4) | q[1]
    BSgate(phi=beta5) | (q[0], q[1])
    BSgate() | (q[0], q[1])
    Sgate(beta6) | q[0]
    Sgate(beta7) | q[1]

    Vgate(gamma0) | q[0]
    Vgate(gamma1) | q[1]
    Dgate(gamma3) | q[0]
    Dgate(gamma4) | q[1]
    BSgate(phi=gamma5) | (q[0], q[1])
    BSgate() | (q[0], q[1])
    Sgate(gamma6) | q[0]
    Sgate(gamma7) | q[1]
    

state = eng.run('tf', cutoff_dim=10, eval=False)
p0 = state.fock_prob([0, 2])
p1 = state.fock_prob([2, 0])
normalization = p0 + p1 + 1e-10
circuit_output = [p0 / normalization, p1 / normalization]
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=circuit_output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
minimize_op = optimizer.minimize(loss)

X_train = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
#Y_train = [1, 1, 0, 0]
Y_train = [[0,1], [0,1], [1,0], [1,0]]
X_test = [[0.25, 0.5], [0.5, 0.25]]
#Y_test = [1, 0]
Y_test = [[0,1], [1,0]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 500
for s in range(steps):
    for i in range(4):
        sess.run(minimize_op, feed_dict={X: X_train[i], y: Y_train[i]})
    if (s % 10 == 0):
        print("{0}% | Loss: {1}".format((s/steps)*100, sess.run(loss, feed_dict={X:X_train[i], y: Y_train[i]})))

print("X       Prediction")
for x in X_test:
    pred = sess.run(circuit_output, feed_dict={X: x})
    if (pred[0] < pred[1]):
        pred = [0,1]
    else:
        pred = [1,0]
    print("{0} || {1}".format(x, pred))