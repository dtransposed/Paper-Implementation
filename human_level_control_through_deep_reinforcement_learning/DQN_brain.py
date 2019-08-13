import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from collections import deque
import random
import numpy as np
from DQN_model import DQN_Model

class DQN_Agent:

    def __init__(self, game_name):

        self.game_name = game_name

        if self.game_name == 'PONG':
            self.action_space = 3

        elif self.game_name == 'BREAKOUT':
            self.action_space = 4

        else:
            ValueError('Error...')

        self.state_shape = (84, 84, 4)
        self.finish_greedy = 5000000
        self.no_frames = 1
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.learning_rate = 0.00025
        self.optimizer = Adam(lr=self.learning_rate)
        self.loss_object = losses.Huber()
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.batch_size = 32
        self.experience = deque(maxlen=50000)
        self.Q_list = []

    def atari_to_dqn(self, atari_action):
        # ATARI PONG: 0: nothing, 1: nothing, 2: up, 3: down, 4: up,5: down
        # ATARI PONG: 0: start, 1: nothing, 2: right, 3: left


        if self.game_name == 'PONG':

            action_up = 0
            action_down = 1
            action_nothing = 2

            dict_atari_to_dqn_pong = {0: action_nothing,
                                      1: action_nothing,
                                      2: action_up,
                                      3: action_down,
                                      4: action_up,
                                      5: action_down}

            return dict_atari_to_dqn_pong[atari_action]

        elif self.game_name == 'BREAKOUT':

            action_right = 0
            action_left = 1
            action_nothing = 2
            action_start = 3


            dict_atari_to_dqn_breakout = {0: action_nothing,
                                          1: action_start,
                                          2: action_right,
                                          3: action_left}

            return dict_atari_to_dqn_breakout[atari_action]

    def dqn_to_atari(self, dqn_action):
        # DQN PONG 0: nothing, 1: up, 2: down
        # DQN BREAKOUT 0: nothing, 1: right , 2: left , 3 start

        if self.game_name == 'PONG':

            action_up = 0
            action_down = 1
            action_nothing = 2

            dict_dqn_to_atari_pong = {action_nothing: 0,
                                      action_up: 2,
                                      action_down: 3}

            return dict_dqn_to_atari_pong[dqn_action]

        elif self.game_name == 'BREAKOUT':

            action_right = 0
            action_left = 1
            action_nothing = 2
            action_start = 3

            dict_dqn_to_atari_breakout = {action_nothing: 0,
                                          action_start: 1,
                                          action_right: 2,
                                          action_left: 3}

            return dict_dqn_to_atari_breakout[dqn_action]

    def uint_to_float(self, state):
        return state.astype('float32') / 255

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = DQN_Model(self.action_space)
        return model

    def save_weights(self, episode):
        self.model.save_weights('DQN-episode-{}'.format(episode))

    def load_weights(self, file_name):
        try:
            self.model.load_weights(file_name)
            print('Model: {} loaded'.format(file_name))
        except:
            print('Starting training from the scratch!')

    def add_to_experience(self, state, action, reward, next_state, done):
        self.experience.append((state, self.atari_to_dqn(action), reward, next_state, done))

    def linear_e_greedy(self):
        if self.no_frames <= self.finish_greedy:
            e = ((self.epsilon_end - self.epsilon_start) / self.finish_greedy) * self.no_frames + self.epsilon_start
        else:
            e = self.epsilon_end
        return e

    def get_action(self, state):

        self.epsilon = self.linear_e_greedy()
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_space)
            return self.dqn_to_atari(action)

        else:
            state = state.astype('float32') / 255
            state = np.expand_dims(state, axis=0)
            q_values = self.model(state)
            action = np.argmax(q_values[0])
            self.Q_list.append(np.max(q_values))
            return self.dqn_to_atari(action)

    @tensorflow.function
    def train_step(self, batch_state, predicted_state):
        with tensorflow.GradientTape() as tape:
            q_values = self.model(batch_state)
            loss = self.loss_object(q_values, predicted_state)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self):

        minibatch = random.sample(self.experience, self.batch_size)

        batch_state = [data[0] for data in minibatch]
        batch_action = [data[1] for data in minibatch]
        batch_reward = [data[2] for data in minibatch]
        batch_next_state = [data[3] for data in minibatch]
        batch_done = [data[4] for data in minibatch]

        batch_state = self.uint_to_float(np.array(batch_state))
        batch_next_state = self.uint_to_float(np.array(batch_next_state))

        predicted_state = self.model.predict(batch_state)
        predicted_next_state = self.target_model.predict(batch_next_state)

        for i in range(self.batch_size):

            if batch_done[i]:
                target = batch_reward[i]
            else:
                target = batch_reward[i] + self.gamma * np.amax(predicted_next_state[i])

            predicted_state[i][batch_action[i]] = target

        loss = self.train_step(batch_state, predicted_state)
        return loss