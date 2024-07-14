import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class Bandit:
    def __init__(self, eps):
        self.k = 10
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        self.epsilon = eps

        # 单轮每个动作价值估计、执行次数
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)

        # n轮的总次数和平均奖励
        # self.time = 0
        # self.average_reward = 0

    # return the index of action
    def action(self):
        if np.random.rand() < self.epsilon:
            # return random from 0 ~ k-1
            return np.random.choice(range(self.k))
        return np.argmax(self.q_estimation)

    def evaluate(self, action):
        reward = np.random.randn() + self.q_star[action]
        return reward

    def update(self, action, reward):
        # self.time += 1
        # self.average_reward += (reward - self.average_reward) / self.time
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

    def reset(self):
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)

    # 这个最优动作指的是平均收益最高的动作，而非每次都比其它动作收益高
    def best_action(self):
        return np.argmax(self.q_star)


def simulate(rounds, steps, bandits):
    rewards = np.zeros((len(bandits), rounds, steps))
    best_action_counts = np.zeros(rewards.shape)
    # 执行多轮，统计某个step下的最优动作比例和平均收益
    for i, bandit in enumerate(bandits):
        for episode in range(rounds):
            bandit.reset()
            for step in range(steps):
                # 在一轮里面，行动一次评估一次，行动根据当前评估得出
                action = bandit.action()
                reward = bandit.evaluate(action)
                bandit.update(action, reward)
                # 统计
                rewards[i, episode, step] = reward
                if action == bandit.best_action():
                    best_action_counts[i, episode, step] = 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_rewards, mean_best_action_counts


def figure_2_1():
    # 生成 10 个标准正态分布随机数，真实q*
    epsilons = [0, 0.5, 0.1, 0.01]
    bandits = [Bandit(eps=eps) for eps in epsilons]
    rounds = 2000
    steps = 1000
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    # print(bandit.average_reward)
    mean_rewards, mean_best_action_counts = simulate(rounds, steps, bandits)
    for eps, rewards in zip(epsilons, mean_rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, mean_best_action_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_2_2.png')
    plt.close()
    # print(mean_rewards[0])
    # print(mean_best_action_counts[0])


figure_2_1()
