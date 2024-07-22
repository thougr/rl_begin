import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class Bandit:
    def __init__(self, eps=0.1, q_star=None, alpha=0.1, armed=10, nonstationary=False, const_step_size=False, q1=0,
                 ucb=False, c=2, use_preference=False, base=0, use_baseline=True):
        self.k = armed
        if q_star is None:
            self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k) + base
        else:
            self.q_star = q_star + base
        self.epsilon = eps

        # 单轮每个动作价值估计、执行次数
        self.q_estimation = np.zeros(self.k) + q1
        self.action_count = np.zeros(self.k)


        # used for exercise 2.5
        self.origin_q_star = self.q_star
        self.alpha = alpha
        self.nonstationary = nonstationary
        self.const_step_size = const_step_size

        # used for ucb
        self.ucb = ucb
        self.c = c

        # used for gradient
        self.use_preference = use_preference
        self.preference = np.zeros(self.k)
        self.use_baseline = use_baseline
        # 单轮的总次数和平均奖励
        self.time = 0
        self.average_reward = 0


    # return the index of action
    def action(self, step):
        if self.ucb:
            c = self.c
            ucb_q = [
                qt + (c * np.sqrt(np.log(step) / nt) if nt != 0 else float('inf'))
                for qt, nt in zip(self.q_estimation, self.action_count)
            ]
            return np.argmax(ucb_q)
        if self.use_preference:
            exp_pref = np.exp(self.preference)
            sum_prob = np.sum(exp_pref)
            exp_pref /= sum_prob
            return np.random.choice(range(self.k), p=exp_pref)

        if np.random.rand() < self.epsilon:
            # return random from 0 ~ k-1
            return np.random.choice(range(self.k))
        return np.argmax(self.q_estimation)

    def evaluate(self, action):
        reward = np.random.randn() + self.q_star[action]
        return reward

    def update(self, action, reward):
        self.time += 1
        self.average_reward += (reward - self.average_reward) / self.time

        self.action_count[action] += 1
        if self.const_step_size:
            self.q_estimation[action] += self.alpha * (reward - self.q_estimation[action])
        else:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        if self.use_preference:
            exp_pref = np.exp(self.preference)
            sum_prob = np.sum(exp_pref)
            exp_pref /= sum_prob
            alpha_delta = self.alpha * (reward - self.average_reward) if self.use_baseline else self.alpha * reward
            self.preference -= (alpha_delta * exp_pref)
            self.preference[action] += alpha_delta
        if self.nonstationary:
            self.q_star = self.q_star + np.random.normal(loc=0.0, scale=0.01, size=self.k)

    def reset(self):
        self.q_star = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        self.q_estimation = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.q_star = self.origin_q_star.copy()
        self.time = 0
        self.average_reward = 0
        self.preference = np.zeros(self.k)

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
                action = bandit.action(step)
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
    epsilons = [0, 0.1, 0.01]
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


def figure_2_3():
    # 生成 10 个标准正态分布随机数，真实q*
    epsilons = [0, 0.1]
    armed = 10
    q_star = np.random.normal(loc=0.0, scale=1.0, size=armed)
    bandits = [Bandit(eps=0, const_step_size=True, q1=5)]
    bandits2 = [Bandit(eps=0.1, const_step_size=True)]
    rounds = 2000
    steps = 1000
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    # print(bandit.average_reward)
    mean_rewards, mean_best_action_counts = simulate(rounds, steps, bandits)
    mean_rewards2, mean_best_action_counts2 = simulate(rounds, steps, bandits2)
    for desc, rewards in zip(["q1=5,eps=0", "q1=0,eps=0.1"], [mean_rewards[0], mean_rewards2[0]]):
        plt.plot(rewards, label=desc)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for desc, counts in zip(["q1=5,eps=0", "q1=0,eps=0.1"], [mean_best_action_counts[0], mean_best_action_counts2[0]]):
        plt.plot(counts, label=desc)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_2_3.png')
    plt.close()
    # print(mean_rewards[0])
    # print(mean_best_action_counts[0])


def figure_2_4():
    # 生成 10 个标准正态分布随机数，真实q*
    epsilons = [0, 0.1]
    armed = 10
    bandits = [Bandit(eps=0, ucb=True, c=1)]
    bandits2 = [Bandit(eps=0.1)]
    rounds = 2000
    steps = 1000
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    # print(bandit.average_reward)
    mean_rewards, mean_best_action_counts = simulate(rounds, steps, bandits)
    mean_rewards2, mean_best_action_counts2 = simulate(rounds, steps, bandits2)
    for desc, rewards in zip(["ucb c=" + str(bandits[0].c), "eps-greedy,eps=0.1"], [mean_rewards[0], mean_rewards2[0]]):
        plt.plot(rewards, label=desc)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for desc, counts in zip(["ucb c=" + str(bandits[0].c), "eps-greedy,eps=0.1"],
                            [mean_best_action_counts[0], mean_best_action_counts2[0]]):
        plt.plot(counts, label=desc)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_2_4.png')
    plt.close()

def figure_2_5():
    # 生成 10 个标准正态分布随机数，真实q*
    epsilons = [0, 0.1]
    armed = 10
    bandits = []
    bandits.append(Bandit(base=4, use_preference=True, use_baseline=True, alpha=0.1))
    bandits.append(Bandit(base=4, use_preference=True, use_baseline=False, alpha=0.1))
    bandits.append(Bandit(base=4, use_preference=True, use_baseline=True, alpha=0.4))
    bandits.append(Bandit(base=4, use_preference=True, use_baseline=False, alpha=0.4))
    rounds = 2000
    steps = 1000
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    # print(bandit.average_reward)
    mean_rewards, mean_best_action_counts = simulate(rounds, steps, bandits)
    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']
    for desc, rewards in zip(labels, mean_rewards):
        plt.plot(rewards, label=desc)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for desc, counts in zip(labels, mean_best_action_counts):
        plt.plot(counts, label=desc)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_2_5.png')
    plt.close()

def figure_2_6():
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda eps: Bandit(eps=eps),
                  lambda alpha: Bandit(use_preference=True, use_baseline=True, alpha=alpha),
                  lambda coef: Bandit(eps=0, ucb=True, c=coef),
                  lambda initial: Bandit(eps=0, const_step_size=True, q1=initial, alpha=0.1)
    ]
    parameters = [np.arange(-7, -1, dtype=np.double),
                  np.arange(-5, 2, dtype=np.double),
                  np.arange(-4, 3, dtype=np.double),
                  np.arange(-2, 3, dtype=np.double)]
    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    rounds = 2000
    steps = 1000
    average_rewards, _ = simulate(rounds, steps, bandits)
    rewards = np.mean(average_rewards, axis=1)
    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i + l], label=label)
        i += l
    plt.xlabel('Parameter($2^x$)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('figure_2_6.png')
    plt.close()


def exercise_2_11():
    labels = [
        # 'epsilon-greedy', 'gradient bandit',
        #       'UCB', 'optimistic initialization',
        'const_step_size']
    generators = [
        # lambda eps: Bandit(eps=eps),
        #           lambda alpha: Bandit(use_preference=True, use_baseline=True, alpha=alpha),
        #           lambda coef: Bandit(eps=0, ucb=True, c=coef),
        #           lambda initial: Bandit(eps=0, const_step_size=True, q1=initial, alpha=0.1),

                  lambda alpha: Bandit(nonstationary=True, const_step_size=True, alpha=alpha)
    ]
    parameters = [
                  # np.arange(-7, -1, dtype=np.double),
                  # np.arange(-5, 2, dtype=np.double),
                  # np.arange(-4, 3, dtype=np.double),
                  # np.arange(-2, 3, dtype=np.double),
                    np.arange(-7, -1, dtype=np.double),

                  ]
    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    rounds = 2000
    steps = 10000
    average_rewards, _ = simulate(rounds, steps, bandits)
    rewards = np.mean(average_rewards, axis=1)
    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i + l], label=label)
        i += l
    plt.xlabel('Parameter($2^x$)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('exercise_2_11.png')
    plt.close()
# figure_2_6()
# figure_2_1()
exercise_2_11()
