import numpy as np

class MultiArmBandit:
    def __init__(self, arms, pulls, e, c):
        self.e = e                          # Epsilon, parameter for e-greedy policy
        self.n = 0                          # Number of executed pulls
        self.c = c                          # Exploration constant
        self.pulls = pulls                  # How many arm pulls we have

        self.arms = arms                    # Number of bandits
        self.action_count = \
            np.zeros(self.arms)             # How many times we have visited a bandit

        self.history =[]
        self.best_action_count = []
        self.selected_actions =[]

        self.true_rewards = \
            [np.random.randn() for _ in range(self.arms)] # Distribution of rewards
        self.rewards = np.zeros(self.arms)                # Mean average score for every bandit
        self.n_rewards = [[] for _ in range(self.arms)]

    def get_reward(self, action):
        """
        Receive a reward from the action
        """
        return self.true_rewards[action]

    def choose_UCB_greedy_action(self):
        """
        The algorithm has probability e to act randomly,
        and probability (1-e) to act according to UCB strategy
        """

        if np.random.uniform(0,1) > self.e:
            ucb = self.rewards + self.c * np.sqrt(np.log(self.n + 1)/(self.action_count + 1))
            return np.argmax(ucb)
        else:
            return np.random.randint(0, self.arms -1)

    def update(self, action, reward):
        self.action_count[action] =+ 1
        self.rewards[action] = self.rewards[action] + 0.1 * (reward - self.rewards[action])
        self.n_rewards[action].append(reward)

        for i in [x for x in range(self.arms) if x!= action]:
            self.n_rewards[i].append(None)




    def play(self):

        for i in range(self.pulls):
            self.n = self.n + 1
            action = self.choose_UCB_greedy_action()
            self.selected_actions.append(action)
            self.best_action_count.append(np.argmax(self.true_rewards) == action)
            reward = self.get_reward(action)
            self.update(action,reward)
            self.history.append(reward)




def run_bandit():
    bandit = MultiArmBandit(arms = 10, pulls = 200, e = 0.05, c = 2)
    bandit.play()
if __name__ == '__main__':
    run_bandit()