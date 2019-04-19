import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import cv2
import numpy as np
from  skimage.color import rgb2gray
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', type=str, default='PongDeterministic-v4')
parser.add_argument('--image_sequence_size', type=int, default=4)
parser.add_argument('--max_no_games', type=int, default= 3000)
parser.add_argument('--finish_greedy', type=int, default= 1000000)
parser.add_argument('--observation_start', type=int, default= 50000)
parser.add_argument('--action_space', type=int, default= 3)
parser.add_argument('--save_every', type=int, default= 50)
parser.add_argument('--print_every', type=int, default = 25)
args = parser.parse_args()



class DQN_Agent:
    def __init__(self, action_space):
        self.state_shape = (84,84,4)
        self.action_space = action_space
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.00025
        self.model = self.build_model()
        self.batch_size = 32
        self.experience = deque(maxlen=50000)

    def atari_to_dqn(self,atari_action):

        action_up = 0
        action_down = 1
        action_nothing = 2
        dict_atari_to_dqn = {0: action_nothing,
                             1: action_nothing,
                             2: action_up,
                             3: action_down,
                             4: action_up,
                             5: action_down}

        return dict_atari_to_dqn[atari_action]

    def dqn_to_atari(self,dqn_action):

        action_up = 0
        action_down = 1
        action_nothing = 2
        dict_dqn_to_atari = {action_nothing: 0,
                             action_up: 2,
                             action_down: 3}

        return dict_dqn_to_atari[dqn_action]



    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.state_shape, kernel_size = 8, strides = 4, filters = 8, activation='relu'))
        model.add(Conv2D(kernel_size=4, strides=2, filters=32, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def add_to_experience(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_space)
            action = self.dqn_to_atari(action)
            return action

        else:
            state = state.astype('float32') / 255
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
            return self.dqn_to_atari(action)

    def train(self):

        ##TODO## implement smart batch training
        minibatch = random.sample(self.experience, self.batch_size)
        states = []
        targets_f = []

        for state, action, reward, next_state, done in minibatch:
            state = state.astype('float32')/255
            state = np.expand_dims(state, axis = 0)
            next_state = next_state.astype('float32')/255
            next_state = np.expand_dims(next_state, axis=0)

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)

            target_f[0][self.atari_to_dqn(action)] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs = 1, verbose = 0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_end:
            self.epsilon *=self.epsilon_decay
        return loss

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(1000)
def preprocess(frame):
    frame = frame[35:195,:]
    frame = rgb2gray(frame)*255
    frame = resize(frame, (84, 84))
    return frame.astype(np.uint8)



env = gym.make(args.game_name)
DQN = DQN_Agent(action_space)
no_frames = 0

for episode in range(dupa):
    total_score = []
    observation_sequence = []
    observation = env.reset()

    while True:
        env.render()

        observation_processed = preprocess(observation)
        observation_sequence.append(observation_processed)
        some_action = 0

        if len(observation_sequence) <= 4:

            observation, reward, done, info = env.step(some_action)

        else:
            observation_sequence.pop(0)
            state = np.stack([observation_sequence[0],
                                           observation_sequence[1],
                                           observation_sequence[2],
                                           observation_sequence[3]],
                                           axis = 2)

            action = DQN.get_action(state)
            new_observation, reward, done, _ = env.step(action)
            total_score.append(reward)
            no_frames = no_frames + 1
            new_observation_processed = preprocess(new_observation)

            new_state = np.stack([observation_sequence[1],
                                           observation_sequence[2],
                                           observation_sequence[3],
                                           new_observation_processed],
                                           axis = 2)


            DQN.add_to_experience(state, action, reward, new_state, done)

            observation = new_observation

            if len(DQN.experience) > 25000 and no_frames%10==0:
                loss = DQN.train()

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, memory len: {}"
                      .format(episode, dupa, np.sum(total_score), DQN.epsilon, len(DQN.experience)))
                break






