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
parser.add_argument('--memory_capacity', type=int, default=15000, help=' Capacity of replay memory')
parser.add_argument('--e_start', type=float, default=1, help=' How e-greedy is our policy')
parser.add_argument('--e_end', type=float, default=0.05, help=' How e-greedy is our policy')
parser.add_argument('--gamma', type=float, default=0.95, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_sequence_size', type=int, default=4)
parser.add_argument('--max_no_games', type=int, default= 20000)
parser.add_argument('--finish_greedy', type=int, default= 1500)
parser.add_argument('--observation_start', type=int, default= 5000)
parser.add_argument('--action_space', type=int, default= 3)
parser.add_argument('--save_every', type=int, default= 100)
args = parser.parse_args()

@tf.function
def train_step(input, observations, model):
    with tf.GradientTape() as tape:
        action, value = DQN(observations)
        loss = loss_object(input, value)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

try:
    with open('current_buffer.data', 'rb') as fp:
        experience = pickle.load(fp)
    print('Loaded exp with len {}'.format(len(experience)))

except:
    print('Start experience from the scratch')
    experience = []

# initialize Q-Network (action-value function)
DQN = MyModel(args.action_space)
try:
    DQN.load_weights('model')
    print('Weights loaded!')
except:
    print('Failed to load weights...')

loss_object = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
train_loss = metrics.Mean(name='train_loss')

# MACHINE: 0: nothing, 1: nothing, 2: up, 3: down, 4: up ,5 :down
dict_machine_to_algorithm = {0: 0,1: 0 , 2: 1, 3: 2, 4: 1, 5: 2}
# ALGO 0: nothing, 1: up, 2: down
dict_algorithm_to_machine = {0: 0, 1: 2, 2: 3}

env = gym.make('PongDeterministic-v4')

total_episode_reward = []


for episode in range(0, args.max_no_games):

    if episode <= args.finish_greedy:
        e = ((args.e_end - args.e_start)/args.finish_greedy) * episode + args.e_start
    else:
        e = args.e_end

    observation_sequence = []
    observation = env.reset()
    total_loss= []

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
                action = env.action_space.sample()



            else:
                action, _ = np.array(DQN(tf.expand_dims(single_observation,0)))
                action = dict_algorithm_to_machine[int(action)]



            new_observation, reward, done, info = env.step(action)


            total_episode_reward.append(reward)

            new_observation_processed = preprocess_observation(new_observation)
            #cv2.imshow('image', new_observation)
            #cv2.waitKey(1)

            new_single_observation = np.stack([observation_sequence[1], observation_sequence[2],
                                              observation_sequence[3], new_observation_processed], axis = 2)

            if len(experience) > args.memory_capacity:
                experience.pop(0)

            experience.append((single_observation, dict_machine_to_algorithm[action], reward, new_single_observation, done))

            if len(experience) > args.observation_start:
                minibatch = random.sample(experience, args.batch_size)

                batch_single_observation = [data[0] for data in minibatch]
                batch_action = [data[1] for data in minibatch]
                batch_reward = [data[2] for data in minibatch]
                batch_new_single_observation = [data[3] for data in minibatch]
                batch_done = [data[4] for data in minibatch]

                batch_y = []

                for i in range(args.batch_size):

                    if batch_done[i] == True:
                        batch_y.append(batch_reward[i])
                    else:
                        _, value = DQN(np.array(batch_new_single_observation))
                        batch_y.append(batch_reward[i] +    args.gamma * value )

                batch_y = np.array(batch_y)
                batch_new_single_observation = np.array(batch_new_single_observation)

                loss = train_step(batch_y, batch_new_single_observation, DQN)

                total_loss.append(loss)
                observation = new_observation

            else:
                pass

            if done:
                 episode = episode + 1

                 if episode % args.save_every == 0:
                    print('Finished episode {} with average reward {}'.format(episode, np.sum(total_episode_reward)/args.save_every))
                    print('Current loss:', np.mean(total_loss))
                    print('Currently the experience buffer is {}'.format(len(experience)))
                    print('e is {}'.format(e))
                    total_episode_reward = []
                    print('Saving weights...')
                    DQN.save_weights('model {}'.format(episode))
                    print('Saving the buffer')
                    with open('outfile.data', 'wb') as fp:
                        pickle.dump(experience, fp)
                        print('Saved exp')
                 break



