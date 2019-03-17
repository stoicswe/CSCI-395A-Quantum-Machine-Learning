import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

iters = 501 # 501
# printout_filename = "log_ghw03_q2_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def roun(pred):
    # if   pred[0] > pred[1]:# and pred[0] > pred[2]:
    #     return "[1,0]" # "[1,0,0]"
    # elif pred[1] > pred[2] and pred[1] > pred[0]:
    #     return "[0,1]" # "[0,1,0]"
    # elif pred[2] > pred[0] and pred[2] > pred[1]:
    #     return "[1,1]" # "[0,1,0]"
    # else:
    #     return str(pred)
    return str(pred)

# def printout(s):
#     try:
#         f = open(printout_filename, "a")
#         f.write
#     print(s)


# xs = [[0.2, 0.4], [0.6, 0.8], [0.4, 0.2], [0.8, 0.6]]
# ys = [[0,1], [0,1], [1,0], [1,0]]

iris = datasets.load_iris()
xs = iris.data[:, :4]  # we only take the first two features.
ys = []
y_orig = iris.target
for y in y_orig:
    if y == 0:
        ys.append([1,0,0])
    elif y == 1:
        ys.append([0,1,0])
    elif y == 2:
        ys.append([0,0,1])
ys = np.array(ys)

# 0 [5.1 3.5 1.4 0.2] [1 0 0]
# 1 [4.9 3.  1.4 0.2] [1 0 0]
# 50 [7.  3.2 4.7 1.4] [0 1 0]
# 51 [6.4 3.2 4.5 1.5] [0 1 0]
# 100 [6.3 3.3 6.  2.5] [0 0 1]
# 101 [5.8 2.7 5.1 1.9] [0 0 1]
xs=[[5.1, 3.5, 1.4, 0.2], [4.9, 3.,  1.4, 0.2], [7.,  3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.3, 3.3, 6.,  2.5], [5.8, 2.7, 5.1, 1.9]]
ys=[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
# for i in range(150):
	# print(i, xs[i], ys[i])
# print('uuuuuuuu')
# print(xs)
# xs=sklearn.preprocessing.normalize(xs)
# for i in range(150):
	# print(i, xs[i], ys[i])
# 0 [0.80377277 0.55160877 0.22064351 0.0315205 ] [1 0 0]
# 1 [0.82813287 0.50702013 0.23660939 0.03380134] [1 0 0]
# 50 [0.76701103 0.35063361 0.51499312 0.15340221] [0 1 0]
# 51 [0.74549757 0.37274878 0.52417798 0.17472599] [0 1 0]
# 100 [0.65387747 0.34250725 0.62274045 0.25947519] [0 0 1]
# 101 [0.69052512 0.32145135 0.60718588 0.22620651] [0 0 1]
# xs = [[0.80377277, 0.55160877, 0.22064351, 0.0315205 ], [0.82813287, 0.50702013, 0.23660939, 0.03380134], [0.76701103, 0.35063361, 0.51499312, 0.15340221],
# [0.74549757, 0.37274878, 0.52417798, 0.17472599],[0.65387747, 0.34250725, 0.62274045, 0.25947519], [0.69052512, 0.32145135, 0.60718588, 0.22620651] ]
# ys = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]


x = tf.placeholder(tf.float32, shape=[4])
y = tf.placeholder(tf.float32, shape=[3])

# SF setup
eng, q = sf.Engine(4)

paramL0 = tf.Variable([0.01]*28)
paramL1 = tf.Variable([0.01]*28)
paramL2 = tf.Variable([0.01]*28)
with eng:
    #LAYER0
    Dgate(x[0]) | q[0]
    Dgate(x[1]) | q[1]
    Dgate(x[2]) | q[2]
    Dgate(x[3]) | q[3]
    # i: 0-3
    Dgate(paramL0[0]) | q[0]
    Dgate(paramL0[1]) | q[1]
    Dgate(paramL0[2]) | q[2]
    Dgate(paramL0[3]) | q[3]
    # i: 0-3!
    BSgate(paramL0[4],paramL0[5]) | (q[0], q[1])
    BSgate() | (q[0], q[1])
    BSgate(paramL0[6],paramL0[7]) | (q[0], q[2])
    BSgate() | (q[0], q[2])
    BSgate(paramL0[8],paramL0[9]) | (q[0], q[3])
    BSgate() | (q[0], q[3])
    BSgate(paramL0[10],paramL0[11]) | (q[1], q[2])
    BSgate() | (q[1], q[2])
    BSgate(paramL0[12],paramL0[13]) | (q[1], q[3])
    BSgate() | (q[1], q[3])
    BSgate(paramL0[14],paramL0[15]) | (q[2], q[3])
    BSgate() | (q[2], q[3])
    # i: 0-3
    Sgate(paramL0[16]) | q[0]
    Sgate(paramL0[17]) | q[1]
    Sgate(paramL0[18]) | q[2]
    Sgate(paramL0[19]) | q[3]
    Kgate(paramL0[20]) | q[0]
    Kgate(paramL0[21]) | q[1]
    Kgate(paramL0[22]) | q[2]
    Kgate(paramL0[23]) | q[3]
    # Vgate(paramL0[24]) | q[0]
    # Vgate(paramL0[25]) | q[1]
    # Vgate(paramL0[26]) | q[2]
    # Vgate(paramL0[27]) | q[3]


    #LAYER1
    # i: 0-3
    # Dgate(paramL1[0]) | q[0]
    # Dgate(paramL1[1]) | q[1]
    # Dgate(paramL1[2]) | q[2]
    # Dgate(paramL1[3]) | q[3]
    # i: 0-3!
    # BSgate(paramL1[4],paramL1[5]) | (q[0], q[1])
    # BSgate() | (q[0], q[1])
    # BSgate(paramL1[6],paramL1[7]) | (q[0], q[2])
    # BSgate() | (q[0], q[2])
    # BSgate(paramL1[8],paramL1[9]) | (q[0], q[3])
    # BSgate() | (q[0], q[3])
    # BSgate(paramL1[10],paramL1[11]) | (q[1], q[2])
    # BSgate() | (q[1], q[2])
    # BSgate(paramL1[12],paramL1[13]) | (q[1], q[3])
    # BSgate() | (q[1], q[3])
    # BSgate(paramL1[14],paramL1[15]) | (q[2], q[3])
    # BSgate() | (q[2], q[3])
    # i: 0-3
    # Sgate(paramL1[16]) | q[0]
    # Sgate(paramL1[17]) | q[1]
    # Sgate(paramL1[18]) | q[2]
    # Sgate(paramL1[19]) | q[3]
    # Kgate(paramL1[20]) | q[0]
    # Kgate(paramL1[21]) | q[1]
    # Kgate(paramL1[22]) | q[2]
    # Kgate(paramL1[23]) | q[3]
    # Vgate(paramL1[24]) | q[0]
    # Vgate(paramL1[25]) | q[1]
    # Vgate(paramL1[26]) | q[2]
    # Vgate(paramL1[27]) | q[3]


    #LAYER2
    # i: 0-3
    # Dgate(paramL2[0]) | q[0]
    # Dgate(paramL2[1]) | q[1]
    # Dgate(paramL2[2]) | q[2]
    # Dgate(paramL2[3]) | q[3]
    # i: 0-3!
    # BSgate(paramL2[4],paramL2[5]) | (q[0], q[1])
    # BSgate() | (q[0], q[1])
    # BSgate(paramL2[6],paramL2[7]) | (q[0], q[2])
    # BSgate() | (q[0], q[2])
    # BSgate(paramL2[8],paramL2[9]) | (q[0], q[3])
    # BSgate() | (q[0], q[3])
    # BSgate(paramL2[10],paramL2[11]) | (q[1], q[2])
    # BSgate() | (q[1], q[2])
    # BSgate(paramL2[12],paramL2[13]) | (q[1], q[3])
    # BSgate() | (q[1], q[3])
    # BSgate(paramL2[14],paramL2[15]) | (q[2], q[3])
    # BSgate() | (q[2], q[3])
    i: 0-3
    # Sgate(paramL2[16]) | q[0]
    # Sgate(paramL2[17]) | q[1]
    # Sgate(paramL2[18]) | q[2]
    # Sgate(paramL2[19]) | q[3]
    # Kgate(paramL2[20]) | q[0]
    # Kgate(paramL2[21]) | q[1]
    # Kgate(paramL2[22]) | q[2]
    # Kgate(paramL2[23]) | q[3]
    # Vgate(paramL2[24]) | q[0]
    # Vgate(paramL2[25]) | q[1]
    # Vgate(paramL2[26]) | q[2]
    # Vgate(paramL2[27]) | q[3]

state = eng.run('tf', cutoff_dim=10, eval=False)

# loss is probability for the Fock state n=1
p0 = state.fock_prob([1,0,0,0])
p1 = state.fock_prob([0,1,0,0])
p2 = state.fock_prob([0,0,1,0])
p3 = state.fock_prob([0,0,0,1])
# p2 = state.fock_prob([1,0,0])
normalization = p0 + p1 + p2 + p3 + 1e-10
# prob = [p0/normalization,p1/normalization]
prob = [p0/normalization,p1/normalization,p2/normalization]

#hypothesis = tf.nn.softmax(prob) #tf.sigmoid(tf.matmul(prob, w1) + b1)
#cross_entropy = -tf.reduce_sum(y * tf.log(hypothesis))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=y)
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	# print(sess.run([hypothesis], feed_dict={x: xs[0], y: ys[0]}))

	# exit()

	for i in range(iters):
		# Train each point
		for j in range(len(xs)):
			sess.run(train_step, feed_dict={x: xs[j], y: ys[j]})

		# if i % 100 == 0:
		#     print("iter "+str(i))
		# if i % (iters-1)/10 == 0:
			# for j in range(len(xs)):
			#     print("x:",xs[j],"y:",ys[j],"loss:",sess.run(cross_entropy, feed_dict={x:xs[j],y:ys[j]}))
			# print()
		if i % 10 == 0:
			print("Iter",i,"loss", sess.run(cross_entropy, feed_dict={x:xs[0],y:ys[0]}) )
			print("Iter",i,"loss", sess.run(cross_entropy, feed_dict={x:xs[2],y:ys[2]}) )
			print("Iter",i,"loss", sess.run(cross_entropy, feed_dict={x:xs[4],y:ys[4]}) )

	for i in range(len(xs)):
		output = sess.run(prob, feed_dict={x: xs[i]})
		print("y:",ys[i],"pred:", output)