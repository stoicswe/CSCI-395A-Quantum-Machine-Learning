import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from env_market import *
from collections import deque
import random

steps = 500

class Agent:
    def __init__(self):
        ##########
        self.memory = deque(maxlen=500000)
        self.apos = 0
        self.bpos = 0
        self.learning_rate = 0.01 #original value: 0.001
        self.gamma = 0.9
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.95
        ##########
        self.x = tf.placeholder(tf.float64, [1,1]) #[[0.]]
        self.y = tf.placeholder(tf.float64, [1,3]) #[[0. 0. 0.]]

        self.d1 = tf.Variable(0.1)
        self.d2 = tf.Variable(0.1)
        self.d3 = tf.Variable(0.1)

        self.s1 = tf.Variable(0.1)
        self.s2 = tf.Variable(0.1)
        self.s3 = tf.Variable(0.1)

        self.v1 = tf.Variable(0.1)
        self.v2 = tf.Variable(0.1)
        self.v3 = tf.Variable(0.1)

        self.bs1 = tf.Variable(0.1)
        self.bs2 = tf.Variable(0.1)
        self.bs3 = tf.Variable(0.1)
        self.bs4 = tf.Variable(0.1)

        self.eng, self.q = sf.Engine(3)
        with self.eng:
            Dgate(self.x[0][0], 0.)  | self.q[0]
            Dgate(self.x[0][0], 0.)  | self.q[1]
            Dgate(self.x[0][0], 0.)  | self.q[2]

            BSgate(self.bs1)        | (self.q[0], self.q[1])
            BSgate()                | (self.q[0], self.q[1])

            BSgate(self.bs2)        | (self.q[0], self.q[2])
            BSgate()                | (self.q[0], self.q[2])

            BSgate(self.bs3)        | (self.q[1], self.q[2])
            BSgate()                | (self.q[1], self.q[2])

            Dgate(self.d1)          | self.q[0]
            Dgate(self.d2)          | self.q[1]
            Dgate(self.d3)          | self.q[2]

            Sgate(self.s1)          | self.q[0]
            Sgate(self.s2)          | self.q[1]
            Sgate(self.s3)          | self.q[2]

            Vgate(self.v1)          | self.q[0]
            Vgate(self.v2)          | self.q[1]
            Vgate(self.v3)          | self.q[2]

        self.state = self.eng.run('tf', cutoff_dim=10, eval=False)
        self.p0 = self.state.fock_prob([0,0,2])
        self.p1 = self.state.fock_prob([0,2,0])
        self.p2 = self.state.fock_prob([2,0,0])
        self.normalization = self.p0 + self.p1 + self.p2 + 1e-10
        self.output = [[self.p0/self.normalization, self.p1/self.normalization, self.p2/self.normalization]]

        self.costf = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.output)
        
        self.sess = tf.Session()
        #self.optimizer = tf.train.GradientDescentOptimizer(0.1)
        self.optimizer = tf.train.AdamOptimizer(0.01)
        self.trainop = self.optimizer.minimize(self.costf)
        self.sess.run(tf.global_variables_initializer())

    def act(self, xi):
        #np.array([[state.A - state.B]])
        return np.argmax(self.sess.run(self.output, feed_dict={self.x:[[xi.A - xi.B]]}))
    
    def predict(self, xi):
        #np.array([[state.A - state.B]])
        return self.sess.run(self.output, feed_dict={self.x:xi})

    def cost(self, X, Y):
        #cross entropy here
        return self.sess.run(self.costf, feed_dict={self.x:X, self.y:Y})
    
    def fit(self, X, Y):
        self.sess.run(self.trainop, feed_dict={self.x:X, self.y:Y})
    
    def remember(self, state, action, reward, next_state, done):
	    self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):

        if (len(self.memory) < batch_size):
            return

        sample_batch = random.sample(self.memory, batch_size)

		#Online training with this sample
        for state, action, reward, next_state, done in sample_batch:
            if done: #End of episode
                target = reward
            else:
                target_f = self.predict([[state.A - state.B]])
                target = reward + self.gamma * np.amax(self.predict([[next_state.A - next_state.B]])[0])
		
                y = np.zeros((1, 3))

                y[:] = target_f[0][:]
                y[0][action] = target

                y_train = []
                y_train.append(y.reshape(3,))
                y_train = np.array(y_train)

                self.fit([[state.A - state.B]], y_train)
                
    def calc_reward(self, cur_state, next_state, action):

        reward = 0.0

        if (action == 0): #Buy A Sell B
            reward =  cur_state.B - next_state.B + next_state.A - cur_state.A
            self.apos = self.apos + 1
            self.bpos = self.bpos - 1
        elif (action == 1):				#Sell A Buy B
            reward =  cur_state.A - next_state.A + next_state.B - cur_state.B
            self.apos = self.apos - 1
            self.bpos = self.bpos + 1
        else:	#Do nothing
            reward = 0

        return reward
    
    def run(self, env):

        cur_st = state()
        nxt_st = state()

        fp = open("reward.txt", "w")
        fp.write("time reward\n")


        for num_episodes in range(steps):

            self.total_reward = 0.0
            env.reset()
            self.replay(100)
			
            self.apos = 0
            self.bpos = 0

			#Gather the first observation
            cur_st, done, msg = env.step()

            for num_steps in range(steps):
				
                act = self.act(cur_st)

                tmp_st = state()
                tmp_st.A = cur_st.A
                tmp_st.B = cur_st.B

                nxt_st, done, msg = env.step()

				#Calculate reward
                act_reward = self.calc_reward(tmp_st, nxt_st, act)

                self.total_reward = self.total_reward + act_reward

                self.remember(tmp_st, act, act_reward, nxt_st, done)

                cur_st.A = nxt_st.A
                cur_st.B = nxt_st.B

				#Episode over, liquidate everything
                if num_steps == steps - 1:
                    self.total_reward = self.total_reward + self.apos * cur_st.A + self.bpos * cur_st.B
                    break
            
            print("Episode: "+str(num_episodes)+" Total Reward: "+str(self.total_reward/1000.0), end='\r')
            s = str(num_episodes)+" "+str(self.total_reward/1000.0)+"\n"
            fp.write(s)
        print()


if __name__ == "__main__":
	np.random.seed(1)
	trade_agent = Agent()
	env = mkt_env()
	Agent.run(trade_agent, env)
