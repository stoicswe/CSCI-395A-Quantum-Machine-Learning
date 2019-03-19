import tensorflow as tf
import numpy


# Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Input
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Linear Model
lm = x * W + b

# Calculating the cost
squared_delta = tf.square(lm-y)
cost = tf.reduce_sum(squared_delta)

# Optimizing the error / cost
optimize = tf.train.GradientDescentOptimizer(0.1)
train = optimize.minimize(cost)

init = tf.global_variables_initializer()
train_X = [0.1, 0.4, 0.6, 0.9]
train_Y = [0.2, 0.5, 0.7, 1.0]

with tf.Session() as sess:
	sess.run(init)

	for i in range(100):
		sess.run(train, {x: train_X, y: train_Y})
		print(sess.run(cost, {x: train_X, y: train_Y}))

	print("Variables W and b: {}".format(sess.run([W, b])))

	for x_i in train_X:
		print("Value for {} is {}".format(x_i, (x_i * sess.run(W) + sess.run(b))))