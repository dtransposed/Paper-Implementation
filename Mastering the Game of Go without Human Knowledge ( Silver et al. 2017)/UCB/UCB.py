import numpy as np
import matplotlib.pyplot as plt


class MultiArmBandit:
    def __init__(self, arms, pulls, e, c, plot):
        self.e = e
        self.c = c
        self.actions = arms
        self.pulls = pulls
        self.t = 0

        self.Q = [0]*self.actions
        self.N = [0]*self.actions
        self.UCB = None

        self.true_probs = [np.random.rand() for _ in range(self.actions)]

        self.selected_action_history = []
        self.selected_optimal_action =[]

        self.plot = plot

    def choose_action(self):

        if np.random.uniform(0, 1) > self.e:
            self.UCB = self.c * np.sqrt(np.log(self.t)/self.N)
            action = np.argmax(self.Q + self.UCB)
        else:
            action = np.random.randint(0, self.actions-1)

        return action

    def get_reward(self, action):
        win_probability = self.true_probs[action]
        if np.random.uniform(0, 1) < win_probability:
            return 1
        else:
            return 0

    def update_N(self, action):

        self.N[action] = self.N[action] + 1

    def update_Q(self, action, reward):

        self.Q[action] = self.Q[action] + (reward - self.Q[action])/self.N[action]

    def plotter(self):

        ind = np.arange(self.actions)
        width = 0.5

        p1 = plt.bar(ind, self.Q, width)
        p2 = plt.bar(ind, self.UCB, width, bottom=self.Q)

        plt.ylabel('Probability')
        plt.xlabel('Actions')
        plt.title('UCB value at iteration {}'.format(self.t))
        plt.xticks(ind, [x for x in range(self.actions)])
        plt.legend((p1[0], p2[0]), ('Estimated Value', 'Uncertainty'))

        plt.savefig('{}.png'.format(self.t))
        plt.clf()

    def plotter_true_probs(self):

        ind = np.arange(self.actions)
        width = 0.5

        plt.bar(ind, self.true_probs, width)

        plt.ylabel('Probability')
        plt.xlabel('Actions')
        plt.title('True reward distribution')
        plt.xticks(ind, [x for x in range(self.actions)])

        plt.savefig('true_probability.png')
        plt.clf()

    def run(self):
        for i in range(self.pulls):

            self.t = self.t + 1

            action = self.choose_action()
            reward = self.get_reward(action)

            self.update_N(action)
            self.update_Q(action, reward)

            self.selected_action_history.append(action)
            self.selected_optimal_action.append(np.argmax(self.true_probs) == action)

            if self.plot and self.t % 5000 == 0:
                self.plotter()
        if self.plot:
            self.plotter_true_probs()
        return self.selected_optimal_action


def UCB1():

    runs = 100
    arms = 10
    pulls = 50000
    e = 0.05
    c = 2
    plot = False

    optimal_action_matrix = np.zeros((runs,pulls))

    for run in range(runs):
        bandit = MultiArmBandit(arms = arms, pulls = pulls, e = e, c = c, plot = plot)
        optimal_action = bandit.run()
        optimal_action_matrix[run, :] = optimal_action

    fraction_optimal_actions = np.mean(optimal_action_matrix, axis = 0) * 100

    plt.plot(fraction_optimal_actions)
    plt.ylabel('% of Optimal Actions')
    plt.xlabel('Iteration')
    plt.title('UCB-1 mean values from {} experiments'.format(runs))
    plt.savefig('UCB-1-results.png')
    plt.clf()


if __name__ == '__main__':
    UCB1()
