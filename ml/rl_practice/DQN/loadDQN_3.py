'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-12 12:12:29
@LastEditors: Jack Huang
@LastEditTime: 2019-09-12 12:12:29
'''
from tensorflow.keras import models, layers, optimizers
import gym 
import numpy as np 

class DQN:
	def __init__(self, obervation_space, action_space):
		self.obervation_space = obervation_space
		self.action_space = action_space
		self.model = self.load_model()

	def load_model(self):
		return models.load_model('./MountainCar-v0-dqn.h5',compile=False)
			

	def action(self, state):
		return np.argmax(self.model.predict(np.array([state]))[0])


def main(env_name):
	# 准备环境
	env = gym.make(env_name)
	observation_space = env.observation_space.shape[0]
	action_space = env.action_space.n 
	print(observation_space, action_space)
	dqn_solver = DQN(observation_space, action_space)
	print('已经拿到武功秘籍!')
	# 表示玩episodes次游戏
	episodes = 30
	# 每一轮游戏里面玩的步数
	steps = 200
	scores = []

	for episode in range(episodes):
		state = env.reset()
		score = 0
		done = False 
		for step in range(steps):
			env.render()
			action = dqn_solver.action(state=state)
			next_state, reward, done, _ = env.step(action)
			reward = 0 if (next_state[0] >=  env.goal_position) else -1
			score += reward
			state = next_state
			if done:
				break

		win_flag = True if reward == 0 else False
		scores.append(score)	

		print('Eps:{} Won:{} Cur:{:.2f} Min:{:.2f} Anv:{:.2f} Max:{:.2f}'\
				.format(episode, win_flag, score, min(scores[-50:]), \
				        np.mean(scores[-50:]), max(scores[-50:])))


if __name__ == "__main__":
	main(env_name="MountainCar-v0")
