import gym
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint
from statistics import median, mean
import pennylane as qml
from pennylane import numpy as np

try:
    dev = qml.device('strawberryfields.fock', wires=4, cutoff_dim=5)    
except:
    print("To run this demo you need to install the strawberryfields plugin...")
	
	
def layer(v):
	qml.Beamsplitter(v[0], v[1], wires=[0,1])
	qml.Beamsplitter(v[2], v[3], wires=[0,2])
	qml.Beamsplitter(v[4], v[5], wires=[0,3])
	qml.Beamsplitter(v[6], v[7], wires=[1,2])
	qml.Beamsplitter(v[8], v[9], wires=[1,3])
	qml.Beamsplitter(v[10], v[11], wires=[2,3])

	qml.Displacement(v[12], v[13],wires=0)
	qml.Displacement(v[14],v[15], wires=1)
	qml.Displacement(v[16], v[17],wires=2)
	qml.Displacement(v[18], v[19],wires=3)

	qml.Squeezing(v[20], v[21], wires=0)
	qml.Squeezing(v[22], v[23], wires=1)
	qml.Squeezing(v[24], v[25],wires=2)
	qml.Squeezing(v[26], v[27],wires=3)

	"""qml.CubicPhase(v[28],wires=0)
	qml.CubicPhase(v[29],wires=1)
	qml.CubicPhase(v[30],wires=2)
	qml.CubicPhase(v[31],wires=3)"""

	qml.Kerr(v[28],wires=0)
	qml.Kerr(v[29],wires=1)
	qml.Kerr(v[30],wires=2)
	qml.Kerr(v[31],wires=3)


	"""qml.Beamsplitter(v[32], v[33], wires=[0,1])
	qml.Beamsplitter(v[34], v[35], wires=[0,2])
	qml.Beamsplitter(v[36], v[37], wires=[0,3])
	qml.Beamsplitter(v[38], v[39], wires=[1,2])
	qml.Beamsplitter(v[40], v[41], wires=[1,3])
	qml.Beamsplitter(v[42], v[43], wires=[2,3])

	qml.Displacement(v[44], v[45],wires=0)
	qml.Displacement(v[46],v[47], wires=1)
	qml.Displacement(v[48], v[49],wires=2)
	qml.Displacement(v[50], v[51],wires=3)

	qml.Squeezing(v[52], v[53], wires=0)
	qml.Squeezing(v[54], v[55], wires=1)
	qml.Squeezing(v[56], v[57],wires=2)
	qml.Squeezing(v[58], v[59],wires=3)

	#qml.CubicPhase(v[60],wires=0)
	#qml.CubicPhase(v[61],wires=1)
	#qml.CubicPhase(v[62],wires=2)
	#qml.CubicPhase(v[63],wires=3)

	qml.Kerr(v[60],wires=0)
	qml.Kerr(v[61],wires=1)
	qml.Kerr(v[62],wires=2)
	qml.Kerr(v[63],wires=3)"""

	# Element-wise nonlinear transformation
	#qml.Kerr(v[4], wires=0)


@qml.qnode(dev)
def quantum_neural_net(var, x):
	qml.Displacement(x[0], 0, wires=0)
	qml.Displacement(x[1], 0, wires=1)
	qml.Displacement(x[2], 0, wires=2)
	qml.Displacement(x[3], 0, wires=3)

	# "layer" subcircuits
	for v in var:
		layer(v)

	return qml.expval.X(0), qml.expval.X(1)



env = gym.make('CartPole-v0')

ind = env.observation_space.shape[0]
adim = env.action_space.n #discrete

#adim = env.action_space.shape[0] # continues


award_set =[]
test_run = 20
best_gen =[]

def softmax(x):
	x = np.exp(x)/np.sum(np.exp(x))
	return x

def lreLu(x):
	alpha=0.2
	return tf.nn.relu(x)-alpha*tf.nn.relu(-x)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def reLu(x):
	return np.maximum(0,x)


	
# initialize weights
np.random.seed(0)
num_layers = 1
# var_init = 0.05 * np.random.randn(num_layers, 16)	

# Function generate initial set of weights and bias
def intial_gen(test_run):
	in_w_list = []
	for i in range(test_run):
		in_w = 3 * np.random.randn(num_layers, 32)
		in_w_list.append(in_w)
	
	generation = [in_w_list]
	return generation

def run_env(env,in_w):
	obs = env.reset()
	award = 0
	for t in range(250):
		out = quantum_neural_net(in_w, obs)
		action = np.argmax(out)

		#print('action', action)

		obs, reward, done, info = env.step(action)
		award += reward 
		if done:
			break
	return award

#Run environment randomly 
def rand_run(env,test_run):
	award_set = []
	generations = intial_gen(test_run)

	for episode in range(test_run):# run env 10 time
		in_w  = generations[0][episode]
		award = run_env(env,in_w)
		award_set = np.append(award_set,award)
	gen_award = [generations, award_set]
	return gen_award  


def mutation(new_dna):

	j = np.random.randint(0,len(new_dna))
	if ( 0 <j < 10): # controlling rate of amount mutation
		for ix in range(j):
			n = np.random.randint(0,len(new_dna)) #random postion for mutation
			new_dna[n] = new_dna[n] + 3 * np.random.rand()

	mut_dna = new_dna

	return mut_dna

def crossover(Dna_list):
	newDNA_list = []
	newDNA_list.append(Dna_list[0])
	newDNA_list.append(Dna_list[1]) 
	
	for l in range(10):  # generation after crassover
		j = np.random.randint(0,len(Dna_list[0]))
		new_dna = np.append(Dna_list[0][:j], Dna_list[1][j:])

		mut_dna = mutation(new_dna)
		newDNA_list.append(mut_dna)

	return newDNA_list

#Generate new set of weigts and bias from the best previous weights and bias

def reproduce(award_set, generations, test_run):
	
	sel = int(test_run/4)

	good_award_idx = award_set.argsort()[-sel:][::-1] # here only best 2 are selected 
	good_generation = []
	DNA_list = []

	new_input_weight = []
	new_input_bias = []

	new_hidden_weight = []

	new_output_weight =[]

	new_award_set = []

	
	#Extraction of all weight info into a single sequence
	for index in good_award_idx:
		
		w1 = generations[0][index]
		dna = w1.reshape(w1.shape[1],-1)
		DNA_list.append(dna) # make 2 dna for good gerneration

	newDNA_list = crossover(DNA_list)

	for newdna in newDNA_list: # collection of weights from dna info
		
		newdna_in_w1 = np.array(newdna[:generations[0][0].size]) 
		new_in_w = np.reshape(newdna_in_w1, (-1,generations[0][0].shape[1]))
		new_input_weight.append(new_in_w)

		new_award = run_env(env, new_in_w) #bias
		new_award_set = np.append(new_award_set,new_award)

	new_generation = [new_input_weight]

	return new_generation, new_award_set


def evolution(env, test_run, n_of_generations):
	gen_award = rand_run(env, test_run)

	current_gens = gen_award[0] 
	current_award_set = gen_award[1]
	best_gen =[]
	A =[]
	max_num = -1000
	best_gene = []

	for n in range(n_of_generations):
		new_generation, new_award_set = reproduce(current_award_set, current_gens, test_run)
		current_gens = new_generation
		current_award_set = new_award_set
		avg = np.average(current_award_set)

		if (avg > max_num):
			max_num = avg
			best_gene = current_gens[0][0]

		
		#if avg > 4:
		#	best_gen = np.array([current_gens[0][0]])
		#	np.save("newtest", best_gen)
		#	best_gene = best_gen

		a = np.amax(current_award_set)
		print("generation: {}, score: {}".format(n+1, a))
		A = np.append(A, a)
	
	np.save("newtest", best_gene)
	Best_award = np.amax(A)

	
	plt.plot(A)
	plt.xlabel('generations')
	plt.ylabel('score')
	plt.grid()

	print('Average accepted score:',mean(A))
	print('Median score for accepted scores:',median(A))
	return plt.show()


#n_of_generations = 10
n_of_generations = 20
evolution(env, test_run, n_of_generations)


param = np.load("newtest.npy")


in_w = param[0]

def test_run_env(env,in_w):
	obs = env.reset()
	award = 0
	for t in range(5):
		#env.render() #thia slows the process
		out = quantum_neural_net(in_w,obs)
		action = np.argmax(out)

		obs, reward, done, info = env.step(action)
		award += reward

		print("time: {}, fitness: {}".format(t, award)) 
		if done:
			break
	return award

print (test_run_env(env, in_w))

