import gym
import numpy as np
import argparse
from utils import preprocess, save_to_file
from DQN_brain import DQN_Agent

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', type=str, default='PongDeterministic-v4')
parser.add_argument('--image_sequence_size', type=int, default=4)
parser.add_argument('--max_no_games', type=int, default=400)
parser.add_argument('--observation_start', type=int, default=50000)
parser.add_argument('--target_frequency_update', type=int, default=1000)
parser.add_argument('--action_space', type=int, default=3)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=25)
parser.add_argument('--checkpoint_name', type=str, default='')
parser.add_argument('--display', type=bool, default=True)
args = parser.parse_args()


if __name__ == "__main__":

    env = gym.make(args.game_name)
    DQN = DQN_Agent(args.action_space)
    DQN.load_weights(args.checkpoint_name)

    total_score = []
    total_loss = []
    list_training_information = []

    for episode in range(1, args.max_no_games):

        observation_sequence = []
        observation = env.reset()

        while True:
            if args.display:
                env.render()

            observation_processed = preprocess(observation)
            observation_sequence.append(observation_processed)
            some_action = 0

            if len(observation_sequence) <= args.image_sequence_size:

                observation, reward, done, info = env.step(some_action)

            else:
                observation_sequence.pop(0)
                state = np.stack([observation_sequence[0],
                                  observation_sequence[1],
                                  observation_sequence[2],
                                  observation_sequence[3]], axis=2)

                action = DQN.get_action(state)
                new_observation, reward, done, _ = env.step(action)
                total_score.append(reward)
                DQN.no_frames = DQN.no_frames + 1
                new_observation_processed = preprocess(new_observation)

                new_state = np.stack([observation_sequence[1],
                                      observation_sequence[2],
                                      observation_sequence[3],
                                      new_observation_processed], axis=2)

                DQN.add_to_experience(state, action, reward, new_state, done)

                observation = new_observation

                if len(DQN.experience) > args.observation_start:
                    loss = DQN.train()
                    total_loss.append(loss)
                    if DQN.no_frames % args.target_frequency_update == 0:
                        DQN.update_target_model()

                if done:
                    if episode % args.print_every == 0:
                        print("Episode: {}/{}, Mean Q-Values: {} Average Reward: {}, Average Loss: {}, Epsilon: {:.4}, Experience length: {}"
                              .format(episode,
                                      args.max_no_games,
                                      np.mean(DQN.Q_list),
                                      np.sum(total_score)/args.print_every,
                                      np.mean(total_loss),
                                      DQN.epsilon,
                                      len(DQN.experience)))
                        list_training_information.append((episode, np.mean(DQN.Q_list), np.sum(total_score)/args.print_every, DQN.epsilon))
                        save_to_file(list_training_information)
                        total_score = []
                        total_loss = []
                        DQN.Q_list = []

                    if episode % args.save_every == 0:
                        DQN.save_weights(episode)
                        print('Weights saved!')

                    break








