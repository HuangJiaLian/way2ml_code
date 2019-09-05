'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-05 16:51:38
@LastEditors: Jack Huang
@LastEditTime: 2019-09-05 20:06:44
'''
import gym
import time 

env = gym.make('MountainCar-v0')
# env = gym.wrappers.Monitor(env, 'records', force=True)
env.reset()
done = False
start_time = time.time() 
while not done:
    action = 2 
    env.step(action)
    env.render()
    if time.time() - start_time > 8:
        break

env.close()

