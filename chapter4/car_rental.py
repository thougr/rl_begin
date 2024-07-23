import numpy as np
from scipy.stats import poisson

# 一个位置的车最大数量(非早晨)
MAX_CARS = 20
# 单天移动车的最大数量
MAX_MOVES = 5
# 早晨允许车的最大数量
MAX_CARS_IN_MORNING = MAX_CARS + MAX_MOVES


class CarRental:
    def __init__(self):
        # 下面变量名后缀为1的为第一个位置，2为第二个位置
        # v(s)
        self.V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        # pi(s)
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
        # p(s'|s,a)
        self.P1 = np.zeros((MAX_CARS_IN_MORNING + 1, MAX_CARS + 1))
        self.P2 = np.zeros((MAX_CARS_IN_MORNING + 1, MAX_CARS + 1))
        # r(s,a)
        self.R1 = np.zeros(MAX_CARS_IN_MORNING + 1)
        self.R2 = np.zeros(MAX_CARS_IN_MORNING + 1)
        self.gamma = 0.9
        self.theta = 0.0000001
        # 租车
        self.lambda_request1 = 3
        self.lambda_request2 = 4
        # self.lambda_request2 = 3
        # 还车
        self.lambda_dropoffs1 = 3
        self.lambda_dropoffs2 = 2
        # self.lambda_dropoffs2 = 3
        self.setup()

    # 提前计算好r(s,a)、p(s'|s,a)
    def setup(self):
        # 第一个位置
        self.load_P_and_R(self.P1, self.R1, self.lambda_request1, self.lambda_dropoffs1)
        # 第二个位置
        self.load_P_and_R(self.P2, self.R2, self.lambda_request2, self.lambda_dropoffs2)
        pass

    def load_P_and_R(self, P, R, lambda_requests, lambda_dropoffs):
        for requests in range(MAX_CARS_IN_MORNING + 1):
            request_prob = poisson.pmf(requests, lambda_requests)
            # 概率会越来越小，达到阈值直接终止
            if request_prob < 0.000001:
                break
            for n in range(MAX_CARS_IN_MORNING + 1):
                R[n] += 10 * min(requests, n) * request_prob
            for dropoffs in range(MAX_CARS_IN_MORNING + 1):
                dropoff_prob = poisson.pmf(dropoffs, lambda_dropoffs)
                if dropoff_prob < 0.000001:
                    break
                for n in range(MAX_CARS_IN_MORNING + 1):
                    satisfied_requests = min(requests, n)
                    new_n = max(0, min(MAX_CARS, n + dropoffs - satisfied_requests))
                    P[n, new_n] += request_prob * dropoff_prob

    def backup_action(self, n1, n2, a):
        # 搬运车数量不能大于原有的数量
        a = max(-n2, min(a, n1))
        # 限制每天最多搬的数量
        a = max(-MAX_MOVES, min(a, MAX_MOVES))
        # s,a
        morning_n1 = n1 - a
        morning_n2 = n2 + a
        # r(s,a)-搬运费用
        result = self.R1[morning_n1] + self.R2[morning_n2] - 2 * abs(a)
        for new_n1 in range(MAX_CARS + 1):
            for new_n2 in range(MAX_CARS + 1):
                result += self.P1[morning_n1, new_n1] * self.P2[morning_n2, new_n2] * self.gamma * self.V[
                    new_n1, new_n2]
        return result

    def policy_eval(self):
        while True:
            delta = 0
            for n1 in range(MAX_CARS + 1):
                for n2 in range(MAX_CARS + 1):
                    old_v = self.V[n1, n2]
                    a = self.policy[n1, n2]
                    self.V[n1, n2] = self.backup_action(n1, n2, a)
                    delta = max(delta, abs(old_v - self.V[n1, n2]))
            if delta < self.theta:
                break

    def greedy_policy(self, n1, n2, epsilon=0.0000000001):
        best_value = -float('inf')
        best_action = None
        for a in range(max(-MAX_MOVES, -n2), min(MAX_MOVES + 1, n1 + 1)):
            this_value = self.backup_action(n1, n2, a)
            if this_value > best_value + epsilon:
                best_value = this_value
                best_action = a
        return best_action

    def show_greedy_policy(self):
        for n1 in range(MAX_CARS + 1):
            print()
            for n2 in range(MAX_CARS + 1):
                print(f"{self.greedy_policy(n1, n2):3}", end="")

    def greedify(self):
        policy_improved = False
        for n1 in range(MAX_CARS + 1):
            for n2 in range(MAX_CARS + 1):
                b = self.policy[n1, n2]
                self.policy[n1, n2] = self.greedy_policy(n1, n2)
                if b != self.policy[n1, n2]:
                    policy_improved = True
        self.show_policy()
        return policy_improved

    def show_policy(self):
        for n1 in range(MAX_CARS + 1):
            print()
            for n2 in range(MAX_CARS + 1):
                print(f"{self.policy[n1, n2]:3}", end="")

    def policy_iteration(self):
        count = 0
        while True:
            print()
            self.policy_eval()
            print(count)
            if not self.greedify():
                break
            count += 1


car_rental = CarRental()
car_rental.policy_iteration()
