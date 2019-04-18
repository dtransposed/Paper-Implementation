import gym
import tensorflow as tf
import cv2
import random
import numpy as np
from tensorflow.keras import Model, losses, optimizers, metrics
from utils import *
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--memory_capacity', type=int, default=100000, help=' Capacity of replay memory')
parser.add_argument('--e_start', type=float, default=1, help=' How e-greedy is our policy')
parser.add_argument('--e_end', type=float, default=0.05, help=' How e-greedy is our policy')
parser.add_argument('--gamma', type=float, default=0.99, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_sequence_size', type=int, default=4)
parser.add_argument('--max_no_games', type=int, default= 10000)
parser.add_argument('--finish_greedy', type=int, default= 2000)
parser.add_argument('--observation_start', type=int, default= 10000)
parser.add_argument('--action_space', type=int, default= 3)
parser.add_argument('--save_every', type=int, default= 5)
args = parser.parse_args()

def linear_e_greedy(e_end, e_start, finish_greedy, episode):
    if episode <= finish_greedy:
        e = ((e_end - e_start)/finish_greedy) * episode + e_start
    else:
        e = e_end

    return e

@tf.function
def train_step(input, observations, actions):
    with tf.GradientTape() as tape:
        value = DQN(observations)
        qreward = tf.reduce_sum(tf.multiply(value, actions), axis=1)
        loss = loss_object(input, qreward)
    gradients = tape.gradient(loss, DQN.trainable_variables)
    optimizer.apply_gradients(zip(gradients, DQN.trainable_variables))
    return loss

try:
    with open('outfile.data', 'rb') as fp:
        experience = pickle.load(fp)
    print('Loaded exp with len {}'.format(len(experience)))

except:
    print('Start experience from the scratch')
    experience = []

# initialize Q-Network (action-value function)
DQN = MyModel(args.action_space)
action_DQN = MyModel(args.action_space)
try:
    DQN.load_weights('model')
    print('Weights loaded!')
except:
    print('Failed to load weights...')

loss_object = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=0.00025)
train_loss = metrics.Mean(name='train_loss')

# MACHINE: 0: nothing, 1: nothing, 2: up, 3: down, 4: up ,5 :down

def atari_to_dqn(atari_action):

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

def dqn_to_atari(dqn_action):

    action_up = 0
    action_down = 1
    action_nothing = 2
    dict_dqn_to_atari = {action_nothing: 0,
                         action_up: 2,
                         action_down: 3}

    return dict_dqn_to_atari[dqn_action]

# ALGO 0: nothing, 1: up, 2: down


env = gym.make('PongDeterministic-v4')

total_episode_reward = []
frame = 1

for episode in range(0, args.max_no_games):

    print('Training episode no: {}'.format(episode))

    e = linear_e_greedy(args.e_end, args.e_start, args.finish_greedy, episode)

    observation_sequence = []
    observation = env.reset()
    total_loss = []
    Q_list = []

    while True:

        observation_processed = preprocess_observation(observation)
        observation_sequence.append(observation_processed)

        some_action = 0

        if len(observation_sequence) <= args.image_sequence_size:

            observation, reward, done, info = env.step(some_action)


        else:
            observation_sequence.pop(0)
            single_observation = np.stack([observation_sequence[0], observation_sequence[1],
                                           observation_sequence[2], observation_sequence[3]], axis = 2)

            if np.random.rand(1) < e:
                action = env.action_space.sample() # action in atari space
            else:
                DQN_input = np.expand_dims(single_observation, axis = 0)
                DQN_input = DQN_input.astype('float32')/255
                values = DQN(DQN_input)
                action = dqn_to_atari(np.argmax(values)) # action in atari space
                Q_list.append(np.max(values))


            new_observation, reward, done, info = env.step(action)

            total_episode_reward.append(reward)

            new_observation_processed = preprocess_observation(new_observation)
            frame = frame + 1
            cv2.imshow('image', new_observation)
            cv2.waitKey(1)

            new_single_observation = np.stack([observation_sequence[1], observation_sequence[2],
                                              observation_sequence[3], new_observation_processed], axis = 2)


            if len(experience) > args.memory_capacity:
                experience.pop(0)

            experience.append((single_observation, atari_to_dqn(action), reward, new_single_observation, done))

            if len(experience) > args.observation_start and frame%100==0:
                minibatch = random.sample(experience, args.batch_size)

                batch_single_observation = [data[0] for data in minibatch]
                batch_action = [data[1] for data in minibatch]
                batch_reward = [data[2] for data in minibatch]
                batch_new_single_observation = [data[3] for data in minibatch]
                batch_done = [data[4] for data in minibatch]

                DQN_batch = action_DQN(np.array(batch_new_single_observation).astype('float32')/255)

                batch_y = []

                for i in range(args.batch_size):

                    if batch_done[i] == True:
                        batch_y.append(batch_reward[i])
                    else:

                        batch_y.append(batch_reward[i] + args.gamma * np.max(DQN_batch[i]))

                batch_y = np.array(batch_y)
                batch_actions =tf.one_hot(batch_action, args.action_space, name="actiononehot")

                loss = train_step(batch_y, np.array(batch_single_observation).astype('float32')/255, batch_actions)

                total_loss.append(loss)
                observation = new_observation

                if frame % 10000 == 0:
                    action_DQN.set_weights(DQN.get_weights())

            else:
                pass

            if done:
                 episode = episode + 1

                 if episode % args.save_every == 0:
                    print('Finished episode {} with average reward {}'.format(episode, np.sum(total_episode_reward)/args.save_every))
                    print('Current loss:', np.mean(total_loss))
                    print('Currently the experience buffer is {}'.format(len(experience)))
                    print('e is {}'.format(e))
                    try:
                        print('average Q is {}'.format(np.max(Q_list)))
                    except:
                        print('')

                    total_episode_reward = []
                    print('Saving weights...')
                    #DQN.save_weights('model {}'.format(episode))
                    print('Saving the buffer')
                    with open('outfile.data', 'wb') as fp:
                        pickle.dump(experience, fp)
                        print('Saved exp')
                 break



