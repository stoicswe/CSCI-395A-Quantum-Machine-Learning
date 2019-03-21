import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import numpy

class QLinearModel:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [1])
        self.Y = tf.placeholder(tf.float32, [1])
        self.alpha0 = tf.Variable(0.1)
        self.alpha1 = tf.Variable(0.1)
        self.alpha2 = tf.Variable(0.1)
        self.alpha3 = tf.Variable(0.1)
        self.alpha4 = tf.Variable(0.1)
        self.alpha5 = tf.Variable(0.1)
        self.alpha6 = tf.Variable(0.1)
        self.alpha7 = tf.Variable(0.1)
        self.alpha8 = tf.Variable(0.1)
        self.eng, self.q = sf.Engine(1)
        with self.eng:
            Dgate(self.X[0], 0.) | self.q[0]

            Dgate(self.alpha1)   | self.q[0]
            Sgate(self.alpha2)   | self.q[0]
            Vgate(self.alpha0)   | self.q[0]

            Dgate(self.alpha4)   | self.q[0]
            Sgate(self.alpha5)   | self.q[0]
            Vgate(self.alpha3)   | self.q[0]

            Dgate(self.alpha7)   | self.q[0]
            Sgate(self.alpha8)   | self.q[0]
            Vgate(self.alpha6)   | self.q[0]
        self.state = self.eng.run('tf', cutoff_dim=10, eval=False)
        self.p0 = self.state.fock_prob([2])
        self.circuit_output = [self.p0*10]
        self.costf = tf.reduce_sum(tf.square(self.circuit_output[0]-self.Y))
        self.optimize = tf.train.GradientDescentOptimizer(0.1)
        self.trainop = self.optimize.minimize(self.costf)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, xi, yi):
        self.sess.run(self.trainop, feed_dict={self.X:[xi], self.Y:[yi]})
    
    def predict(self, xi):
        return self.sess.run(self.circuit_output, feed_dict={self.X:[xi]})
    
    def cost(self, xi, yi):
        return self.sess.run(self.costf, feed_dict={self.X:[xi], self.Y:[yi]})
    

myModel = QLinearModel()

train_X = [0.1, 0.4, 0.6, 0.9]
train_Y = [0.2, 0.5, 0.7, 1.0]

for i in range(1000):
    for i in range(len(train_X)):
        myModel.train(train_X[i], train_Y[i])
    print(myModel.cost(train_X[0], train_Y[0]))

for x_i in train_X:
	print("Value for {} is {}".format(x_i, myModel.predict(x_i)))