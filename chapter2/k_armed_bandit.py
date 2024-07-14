import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class Bandit:
    def __init__(self, eps, q_star=None, alpha=0.1, armed=10, nonstationary=False, const_step_size=False):
        self.k = armed
        if q_star is None:
            self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        else:
            self.q_star = q_star
        self.epsilon = eps

        # 单轮每个动作价值估计、执行次数
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)

        # n轮的总次数和平均奖励
        # self.time = 0
        # self.average_reward = 0

        # used for exercise 2.5
        self.origin_q_star = self.q_star
        self.alpha = alpha
        self.nonstationary = nonstationary
        self.const_step_size = const_step_size

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
        if self.const_step_size:
            self.q_estimation[action] += self.alpha * (reward - self.q_estimation[action])
        else:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        if self.nonstationary:
            self.q_star = self.q_star + np.random.normal(loc=0.0, scale=0.01, size=self.k)

    def reset(self):
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.q_star = self.origin_q_star.copy()

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

def exercise_2_5():
    # 生成 10 个标准正态分布随机数，真实q*
    epsilons = [0.1]
    armed = 10
    q_star = np.random.normal(loc=0.0, scale=1.0, size=armed)
    bandits = [Bandit(eps=eps, q_star=q_star, nonstationary=True, const_step_size=False) for eps in epsilons]
    bandits2 = [Bandit(eps=eps, q_star=q_star, nonstationary=True, const_step_size=True) for eps in epsilons]
    rounds = 2000
    steps = 10000
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    # print(bandit.average_reward)
    mean_rewards, mean_best_action_counts = simulate(rounds, steps, bandits)
    mean_rewards2, mean_best_action_counts2 = simulate(rounds, steps, bandits2)
    for desc, rewards in zip(["average_step", "const_step"], [mean_rewards[0], mean_rewards2[0]]):
        plt.plot(rewards, label=desc)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for desc, counts in zip(["average_step", "const_step"], [mean_best_action_counts[0], mean_best_action_counts2[0]]):
        plt.plot(counts, label=desc)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_e_2_5.png')
    plt.close()
    # print(mean_rewards[0])
    # print(mean_best_action_counts[0])

exercise_2_5()




