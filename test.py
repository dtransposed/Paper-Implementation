import gym
import cv2
env = gym.make('PongDeterministic-v4')
action = 0 # modify this!
o = env.reset()
for i in range(20): # repeat one action for five times
    o = env.step(action)[0]
    cv2.imshow('image', o)
    cv2.waitKey(100)
