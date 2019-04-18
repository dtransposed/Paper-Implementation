import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np
EPISODES = 1000
MEMORY = 2000

class DQN_Agent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.batch_size = 32
        self.experience = deque(maxlen=2000)



    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def add_to_experience(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train(self):

        ##TODO## implement smart batch training
        minibatch = random.sample(self.experience, self.batch_size)
        states = []
        targets_f = []

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)

            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs = 1, verbose = 0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_end:
            self.epsilon *=self.epsilon_decay
        return loss



env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape[0]
action_space = 2
DQN = DQN_Agent(state_shape, action_space)
is_done = False

for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_shape])
    for time in range(500):
        env.render()
        action = DQN.get_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10
        next_state = np.reshape(next_state, [1, state_shape])
        DQN.add_to_experience(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(episode, EPISODES, time, DQN.epsilon))
            break
        if len(DQN.experience) > DQN.batch_size:
            loss = DQN.train()
            # Logging training loss every 10 timesteps





