'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-05 20:33:19
@LastEditors: Jack Huang
@LastEditTime: 2019-09-05 20:43:18
'''

import gym
import time 
import numpy as np 

env = gym.make('MountainCar-v0')
env = gym.wrappers.Monitor(env, 'records', force=True)
env.reset()
done = False
action = np.random.randint(0,3)
while not done:
    new_state, reward, done, _ = env.step(action)
    env.render()
    if new_state[-1] < 0:
        action = 0 # 向左
    elif new_state[-1] > 0:
        action = 2 # 向右
    else:
        action = 1
    
env.close()
