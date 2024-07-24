import numpy as np
import matplotlib.pyplot as plt

TARGET = 100
class Gambler:
    def __init__(self, win_prob=0.45, epsilon=1e-8):
        self.V = np.zeros(TARGET + 1)
        self.V[TARGET] = 1
        self.p = win_prob
        self.epsilon = epsilon

    def backup_action(self, s, a):
        win = self.p * self.V[s + a]
        lose = (1 - self.p) * self.V[s - a]
        return win + lose

    def value_iteration(self):
        while True:
            delta = 0
            for s in range(1, TARGET):
                old_value = self.V[s]
                self.V[s] = max(self.backup_action(s, a) for a in range(1, min(s, 100 - s) + 1))
                delta = max(delta, abs(old_value - self.V[s]))
            if delta < self.epsilon:
                break
        return [(i, self.V[i]) for i in range(0, TARGET + 1)]

    def policy(self, s, epsilon=1e-5):
        best_value = -1
        best_action = []
        for a in range(1, min(s, 100 - s) + 1):
            this_value = self.backup_action(s, a)
            if abs(this_value - best_value) < epsilon:
                best_action.append(a)
            elif this_value > best_value:
                best_action = []
                best_value = this_value
                best_action.append(a)
        return best_action

    def get_policy(self):
        return [self.policy(s) for s in range(TARGET + 1)]

# gambler = Gambler()
# <0.5 往25、50、75这三个极值点靠，收益是最大的
gambler = Gambler(win_prob=0.25)
# 0.5是怎么投都无所谓
# gambler = Gambler(win_prob=0.5)
# >0.5是初期保守，到中期旧随便了（当然不能太大）
# gambler = Gambler(win_prob=0.55)
values = gambler.value_iteration()
policy = gambler.get_policy()

# 打印结果
print("Values:", values)
print("Policy:", policy)

# 绘制值函数
# 将values拆分为x和y
values_x = [v[0] for v in values]
values_y = [v[1] for v in values]
plt.figure(figsize=(12, 6))
plt.plot(values_x, values_y, label='Value Function')
plt.xlabel('Stake')
plt.ylabel('Value')
plt.title('Value Function for Gambler\'s Problem')
plt.legend()
plt.grid()
plt.show()

# 绘制策略
plt.figure(figsize=(12, 6))
plt.plot(range(TARGET + 1), [max(actions) if actions else 0 for actions in policy], label='max stake Policy')
plt.plot(range(TARGET + 1), [min(actions) if actions else 0 for actions in policy], label='min stake Policy')
middle = []
for actions in policy:
    if actions:
        if len(actions) == 3:
            middle.append(actions[1])
        else:
            middle.append(min(actions))
    else:
        middle.append(0)
plt.plot(range(TARGET + 1), middle, label='middle stake Policy')
plt.xlabel('Stake')
plt.ylabel('Optimal Action')
plt.title('Optimal Policy for Gambler\'s Problem')
plt.legend()
plt.grid()
plt.show()