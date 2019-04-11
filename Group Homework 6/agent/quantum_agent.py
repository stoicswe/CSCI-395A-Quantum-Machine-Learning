import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

import random
import os
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.firstIter = True

        self.x = tf.placeholder(tf.float64, [1,4])
        self.y = tf.placeholder(tf.float64, [1,3])

        self.d1 = tf.Variable(0.1)
        self.d2 = tf.Variable(0.1)
        self.d3 = tf.Variable(0.1)
        self.d4 = tf.Variable(0.1)

        self.s1 = tf.Variable(0.1)
        self.s2 = tf.Variable(0.1)
        self.s3 = tf.Variable(0.1)
        self.s4 = tf.Variable(0.1)

        self.v1 = tf.Variable(0.1)
        self.v2 = tf.Variable(0.1)
        self.v3 = tf.Variable(0.1)
        self.v4 = tf.Variable(0.1)

        self.bs1 = tf.Variable(0.1)
        self.bs2 = tf.Variable(0.1)
        self.bs3 = tf.Variable(0.1)
        self.bs4 = tf.Variable(0.1)
        self.bs5 = tf.Variable(0.1)
        self.bs6 = tf.Variable(0.1)
        self.bs7 = tf.Variable(0.1)
        self.bs8 = tf.Variable(0.1)
        self.bs9 = tf.Variable(0.1)

        self.bs10 = tf.Variable(0.1)
        self.bs11 = tf.Variable(0.1)
        self.bs12 = tf.Variable(0.1)
        self.bs13 = tf.Variable(0.1)
        self.bs14 = tf.Variable(0.1)
        self.bs15 = tf.Variable(0.1)
        self.bs16 = tf.Variable(0.1)
        self.bs17 = tf.Variable(0.1)
        self.bs18 = tf.Variable(0.1)


        self.eng, self.q = sf.Engine(4)
        with self.eng:
            Dgate(self.x[0][0], 0.)  | self.q[0]
            Dgate(self.x[0][1], 0.)  | self.q[1]
            Dgate(self.x[0][2], 0.)  | self.q[2]
            Dgate(self.x[0][3], 0.)  | self.q[3]

            BSgate(self.bs1)        | (self.q[0], self.q[1])
            BSgate()                | (self.q[0], self.q[1])
            BSgate(self.bs2)        | (self.q[0], self.q[2])
            BSgate()                | (self.q[0], self.q[2])
            BSgate(self.bs3)        | (self.q[0], self.q[3])
            BSgate()                | (self.q[0], self.q[3])

            BSgate(self.bs10)        | (self.q[1], self.q[2])
            BSgate()                 | (self.q[1], self.q[2])
            BSgate(self.bs11)        | (self.q[1], self.q[3])
            BSgate()                 | (self.q[1], self.q[3])

            BSgate(self.bs18)        | (self.q[2], self.q[3])
            BSgate()                 | (self.q[2], self.q[3])

            Dgate(self.d1)          | self.q[0]
            Dgate(self.d2)          | self.q[1]
            Dgate(self.d3)          | self.q[2]
            Dgate(self.d4)          | self.q[3]

            Sgate(self.s1)          | self.q[0]
            Sgate(self.s2)          | self.q[1]
            Sgate(self.s3)          | self.q[2]
            Sgate(self.s4)          | self.q[3]

            Vgate(self.v1)          | self.q[0]
            Vgate(self.v2)          | self.q[1]
            Vgate(self.v3)          | self.q[2]
            Vgate(self.v4)          | self.q[3]

        self.state = self.eng.run('tf', cutoff_dim=10, eval=False)
        self.p0 = self.state.fock_prob([0,0,0,2])
        self.p1 = self.state.fock_prob([0,0,2,0])
        self.p2 = self.state.fock_prob([0,2,0,0])
        self.p3 = self.state.fock_prob([2,0,0,0])
        self.normalization = self.p0 + self.p1 + self.p2 + self.p3 + 1e-10
        self.output = [[self.p0/self.normalization, self.p1/self.normalization, self.p2/self.normalization]]

        self.costf = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.output)
        
        self.sess = tf.Session()
        #self.optimizer = tf.train.GradientDescentOptimizer(0.1)
        self.optimizer = tf.train.AdamOptimizer(0.01)
        self.trainop = self.optimizer.minimize(self.costf)
        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        rand_val = np.random.rand()
        if not self.is_eval and rand_val <= self.epsilon:
            return random.randrange(self.action_size)
        if(self.firstIter):
            self.firstIter = False
            return 1
        options = self.sess.run(self.output, feed_dict={self.x:state})
        return np.argmax(options[0])
    
    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory.popleft())

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
	            target = reward + self.gamma * np.amax(self.sess.run(self.output, feed_dict={self.x:next_state})[0])

            target_f = self.sess.run(self.output, feed_dict={self.x:state})
            target_f[0][action] = target
            #fit the quantum neural network
            self.sess.run(self.trainop, feed_dict={self.x:state, self.y:target_f})
            #self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

