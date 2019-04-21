import gym
import numpy as np
import argparse
from utils import preprocess
from DQN_brain import DQN_Agent
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', type=str, default='PongDeterministic-v4')
parser.add_argument('--image_sequence_size', type=int, default=4)
parser.add_argument('--action_space', type=int, default=3)
parser.add_argument('--checkpoint_name', type=str, default='DQN-episode-300')
parser.add_argument('--display', type=bool, default=True)
args = parser.parse_args()


if __name__ == "__main__":
    images = []
    frame_no = 0
    env = gym.make(args.game_name)
    DQN = DQN_Agent(args.action_space)
    DQN.load_weights(args.checkpoint_name)

    while True:

        observation_sequence = []
        observation = env.reset()

        while True:
            if args.display:
                env.render()
                if frame_no % 2 ==0 and frame_no != 0:
                    env.env.ale.saveScreenPNG('test_image {}.png'.format(frame_no))
                    images.append(imageio.imread('test_image {}.png'.format(frame_no)))

            observation_processed = preprocess(observation)
            observation_sequence.append(observation_processed)
            frame_no = frame_no + 1
            some_action = 0

            if len(observation_sequence) <= args.image_sequence_size:

                observation, reward, done, info = env.step(some_action)

            else:
                observation_sequence.pop(0)
                state = np.stack([observation_sequence[0],
                                  observation_sequence[1],
                                  observation_sequence[2],
                                  observation_sequence[3]], axis=2)

                state = DQN.uint_to_float(np.array(state))
                state = np.expand_dims(state, axis=0)
                action = np.argmax(DQN.model(state))



                new_observation, reward, done, _ = env.step(DQN.dqn_to_atari(action))

                new_observation_processed = preprocess(new_observation)

                new_state = np.stack([observation_sequence[1],
                                      observation_sequence[2],
                                      observation_sequence[3],
                                      new_observation_processed], axis=2)

                observation = new_observation

                if done:
                    imageio.mimsave('pong_gif/movie.gif', images)
                    break
