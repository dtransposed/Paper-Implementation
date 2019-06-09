import numpy as np
import matplotlib.pyplot as plt

def get_reward(action, ground_truth_probs):
    '''
    Reward function which tells us if we get a reward for
    choosing an action or not, given ground truth machine probabilities.
    '''
    return 1 if np.random.uniform(0, 1) < ground_truth_probs[action] else 0

def define_ground_truth_probs(no_bandits):
    '''
    Function which randomly initializes ground truth
    probabilities.
    '''
    biases = [1.0 / k for k in range(5, 5 + no_bandits)]
    #biases = [random_sign() * k for k in biases]
    gt_probs = [0.5 + b for b in biases]
    print('Initial Ground Truth probabilities: {}'.format(gt_probs))
    return gt_probs

def random_sign():
    return 1 if np.random.uniform(0, 1) < 0.5 else -1

# problem setup
no_bandits = 4
no_episodes = 100000


gt_probs = define_ground_truth_probs(no_bandits)

# creating placeholders to store algorithm history
history_pull_count = np.zeros((no_episodes, no_bandits))
history_estimation = np.zeros((no_episodes, no_bandits))
history_reward = np.zeros(no_episodes)
history_regret = np.zeros(no_episodes)

# optimal action
optimal_action = np.argmax(gt_probs)

# START TRAINING #

pull_count = np.zeros(no_bandits)
estimation = np.zeros(no_bandits)

for episode in range(no_episodes):
    # select an action according to UCB1 formulation
    current_action = np.argmax(estimation + np.sqrt(2 * np.log(episode + 1) / (pull_count + 1)))
    # sample reward from reward function
    reward = get_reward(current_action, gt_probs)

    # update counts
    pull_count[current_action] = pull_count[current_action] + 1
    # update action-value (mean reward estimate)
    estimation[current_action] = estimation[current_action] + (1 / (pull_count[current_action] + 1)) * (reward - estimation[current_action])
    # compute regret for the current episode
    regret = (gt_probs[optimal_action] - gt_probs[current_action])
    print(regret)

    # update history and update cumulative regret / reward
    history_pull_count[episode, :] = pull_count
    history_estimation[episode, :] = estimation
    history_reward[episode] = history_reward[episode - 1] + reward
    history_regret[episode] = history_regret[episode - 1] + regret

print('Ground Truth')
print(gt_probs)
print('Expected ')
print(history_estimation[-1])
plt.plot(history_regret)
plt.plot(history_reward)

plt.show()




