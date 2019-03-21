import tensorflow as tf
import numpy

class LinearModel:
    def __init__(self, sess):
        self.W = tf.Variable([.3], tf.float32)
        self.b = tf.Variable([-.3], tf.float32)
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.function = self.x * self.W + self.b
        self.costf = tf.reduce_sum(tf.square((self.x * self.W + self.b)-self.y))
        self.sess = sess
        self.optimizer = tf.train.GradientDescentOptimizer(0.1)
        self.trainop = self.optimizer.minimize(self.costf)
        self.sess.run(tf.global_variables_initializer())

    def predict(self, xi):
        return self.sess.run(self.function, feed_dict={self.x:xi})

    def cost(self, X, Y):
        return self.sess.run(self.costf, feed_dict={self.x:X, self.y:Y})
    
    def train(self, X, Y):
        self.sess.run(self.trainop, feed_dict={self.x:X, self.y:Y})
    

myModel = LinearModel(tf.Session())

init = tf.global_variables_initializer()
train_X = [0.1, 0.4, 0.6, 0.9]
train_Y = [0.2, 0.5, 0.7, 1.0]

for i in range(100):
    myModel.train(train_X, train_Y)
    print(myModel.cost(train_X, train_Y))

for x_i in train_X:
	print("Value for {} is {}".format(x_i, myModel.predict(x_i)))