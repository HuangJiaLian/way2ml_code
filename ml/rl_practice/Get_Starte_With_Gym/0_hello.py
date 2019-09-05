'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-05 16:49:57
@LastEditors: Jack Huang
@LastEditTime: 2019-09-05 16:49:57
'''
import gym
import time 
env = gym.make('MountainCar-v0')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
env.reset()
env.render()
time.sleep(5)
env.close()
