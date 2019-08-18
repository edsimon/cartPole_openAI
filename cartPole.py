
import gym
import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import pickle
import tensorflow as tf
import operator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.logging.set_verbosity(tf.logging.ERROR)

env = gym.make('CartPole-v0')
goal_steps = 10000



def main() : 
	intro_message()
	samples, percent = get_settings()
	training_data = get_random_samples(samples, percent)
	model = train_network(training_data)
	demo(model)

	while (run_again()):
		samples, percent = get_settings()		
		training_data = get_random_samples(samples, percent, model=True)
		model = train_network(training_data)		
		demo(model)





def get_random_samples(samples, percent, model=False) :
	if model : 
		model = neural_model()
		model.load("model.tfl")
	sample	= samples
	data	= []
	scores	= []
	games = []
	samples_each_bar = int(samples/50)
	print("Generating samples!")
	for _ in range(sample) : 
		game_data = []
		previous_observation = []
		score = 0
		env.reset()
		for t in range(1000) :
			#env.render()

			# Get 20% random moves to not overfit model.
			chance = random.randrange(0,6)
			if (chance == 0) : 
				action = random.randrange(0,2)
				observation, reward, done, info = env.step(action)

			else : 
				if (len(previous_observation)==0):
					action = random.randrange(0,2)
				elif (model):
					action = np.argmax(model.predict(previous_observation.reshape(-1,len(previous_observation),1))[0])
				else : 
					action = random.randrange(0,2)

				observation, reward, done, info = env.step(action)

				if (len(previous_observation) > 0) :
					game_data.append([previous_observation, action])

				previous_observation = observation

			if done :
				score = t
				break

		scores.append(score)
		if ( len(scores) % samples_each_bar == 1 and len(scores) != 1) : 
			print("|",end="")
		games.append(game_data)

	data = get_highest_score(games, scores, percent)
	data = clean_data(data)
	
	scores.sort()
	scores = scores[int(len(scores)*percent) : ]

	print("\n")
	print("Number of generated games: ", len(scores))
	print('Average accepted score:', mean(scores))
	print('Median score for accepted scores:', median(scores))
	print(Counter(scores))

	env.close()
	np.save("random_sample.npy", data)

	return data


def neural_model() :
	tf.reset_default_graph()
	network = input_data(shape=[None, 4, 1], dtype=tf.float32, name='input')
	
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model


def train_network(training_data) :
	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]

	model = neural_model()

	model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')

	# Saving model
	model.save("model.tfl")

	return model


def get_highest_score(data, scores, percent) :
	# Returns the top 20% of all scores generated
	clean_data = []
	list_of_tuples = list(zip(scores, data))
	list_of_tuples.sort(key = operator.itemgetter(0))
	list_of_tuples = list_of_tuples[int(len(list_of_tuples)*percent) : ]
	clean_data = [ i[1] for i in list_of_tuples ]
	return clean_data


def clean_data(data): 
	# Turns our action into a [0,1] or [1,0] and concatinate the result
	clean_data = []
	for game in data : 
		for d in game :
			if (d[1] == 1):
				output = [0 ,1]

			elif d[1] == 0:
				output = [1 ,0]
			

			clean_data.append([d[0], output])
	return clean_data


def demo(model) :
	input("Press Enter to run a demo of your AI")
	scores = []
	choices = []
	for each_game in range(5):
		score = 0
		game_memory = []
		prev_obs = []
		env.reset()
		for _ in range(goal_steps):
			env.render()

			if len(prev_obs)==0:
				action = random.randrange(0,2)
			else:
				action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

			choices.append(action)
					
			new_observation, reward, done, info = env.step(action)
			prev_obs = new_observation
			game_memory.append([prev_obs, action])
			score+=reward
			if done: 
				break

		scores.append(score)

	print('Average Score:',sum(scores)/len(scores))
	print('choice 1: {}  choice 0: {}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))


def intro_message(): 
	print("\n"*100)
	print("Welcome to CartPole-v0")	
	print("This is a simulation where we first of generate")
	print("a number of samples. Then we start of by choosing ")
	print("how many percentage of the top performing samples ")
	print("that we will be using to train our model. The lower ")
	print("the better. After the first run you will get")
	print("a choice if you like to retrain newsamples based")
	print("of your model to improve it.\n")


def get_settings():
	print("Start by choosing some of the settings.\n")
	samples = int(input("How many samples do you want to train?\n"))
	percent = int(input("How many percent of the samples do you want to use? [0-100]\n"))
	percent = 1-float(percent/100)
	print("Ok, Let´s go!\n")
	return samples, percent


def run_again() : 
	print("\n"*3)
	print("It´s usually better to run several learning iterations.")
	print("This comes with a cost that it is slower\n")
	run_again = input("Do you want to run one more, press y?")
	if(run_again == "y" or run_again == "Y"):
		return True
	else :
		return False


main()
