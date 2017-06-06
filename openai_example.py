import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter

LR=1e-3
env=gym.make('CartPole-v0')
env.reset()
goal_steps=500
score_requirements=50
initial_games=10000

def random_game():
	for episodes in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action=env.action_space.sample()
			observation,reward,done,info=env.step(action)
			if done:
				break


#random_game()

def init_population():
	training_data=[]
	scores=[]
	accepted_scores=[]
	for i in range(initial_games):
		score=0
		game_memory= []
		previous_obs= []
		for i in range(goal_steps):
			action=random.randrange(0,2)
			observation,reward,done,info=env.step(action)

			if len(previous_obs) > 0:
				game_memory.append([previous_obs,action])

			previous_obs=observation
			score += reward
			if done:
				break


		if score >= score_requirements:
			accepted_scores.append(score)

			for data in game_memory:
				#print(game_memory[0])
				if data[1] == 1:
						output=[0,1]

				elif data[1] == 0:
					output=[1,0]

				training_data.append([data[0],output])


		env.reset()
		scores.append(score)


	trianing_data_save=np.array(training_data)
	np.save('saved.npy',trianing_data_save)

	print('Average Scores',mean(accepted_scores))
	print('Median Scores',median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data


init_population()
def neural_network(input_size):
	network=input_data(shape=[None,input_size,1],name='input')
	
	network=fully_connected(network,128,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,256,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,512,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,128,activation='relu')
	network=dropout(network,0.8)

	network=fully_connected(network,2,activation='softmax')
	network=regression(network,optimizer = 'adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')

	model=tflearn.DNN(network,tensorboard_dir='log')

	return model


def train_model(training_data,model=False):
	X=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y=[i[1] for i in training_data]

	if not model:
		model=neural_network(input_size=len(X[0]))

	model.fit({'input':X},{'targets':y},n_epoch=4,snapshot_step=500,show_metric=True,run_id='openaistuff')

	return model

training_data=init_population()
model=train_model(training_data)

scores = []
choices = []

for game_count in range(100):
	score=0
	game_memory=[]
	previous_obs=[]
	env.reset()

	for i in range(goal_steps):
		env.render()
		if len(previous_obs) == 0:
			action=random.randrange(0,2)

		else :
			action=np.argmax(model.predict(previous_obs.reshape(-1,len(previous_obs),1))[0])
		choices.append(action)

		new_observation,reward,done,info=env.step(action)
		previous_obs=new_observation
		game_memory.append([new_observation,action])
		score += reward
		if done:
			break
	scores.append(score)

print('Average Scores',sum(scores)/len(scores))
print('Choice 1 : {}, Choice 0: {}'.format(float(choices.count(1))/float(len(choices)),
	   float(choices.count(0))/float(len(choices))))







