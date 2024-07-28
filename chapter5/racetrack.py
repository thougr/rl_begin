import math
import random

import numpy as np
from matplotlib import pyplot as plt

ROWS = 32
COLS = 17
MAX_VELOCITY = 4

actions = [
    (0, 0),
    (1, 1),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
ACTION_TYPE = len(actions)
track = np.zeros(shape=(ROWS, COLS), dtype=np.uint8)
track[31, 0:3] = 1
track[30, 0:2] = 1
track[29, 0:2] = 1
track[28, 0] = 1
track[0:18, 0] = 1
track[0:10, 1] = 1
track[0:3, 2] = 1
track[0:26, 9:] = 1
track[25, 9] = 0
start_cols = list(range(3, 9))  # start line columns
fin_cells = [(i, COLS - 1) for i in range(26, ROWS)]  # finish cells

# 表示当前的episode是否结束
END = 1
CONTINUE = 0

MIN_VALUE = -999

#表示是不是破撞到边界了
RESTART_ROUND = 1
NOT_RESTART_ROUND = 0

class Racetrack(object):
    def __init__(self, eps=0.1):
        self.V = np.full(shape=(ROWS, COLS, MAX_VELOCITY + 1, MAX_VELOCITY + 1), fill_value=MIN_VALUE)
        self.Q = np.full(shape=(ROWS, COLS, MAX_VELOCITY + 1, MAX_VELOCITY + 1, ACTION_TYPE), fill_value=MIN_VALUE)
        self.N = np.zeros(shape=(ROWS, COLS, MAX_VELOCITY + 1, MAX_VELOCITY + 1, ACTION_TYPE))
        self.policy = np.ones(shape=(ROWS, COLS, MAX_VELOCITY + 1, MAX_VELOCITY + 1), dtype=np.uint8)
        self.epsilon = eps


    def action(self, x, y, v_x, v_y):
        if random.random() < 0.1:
            # zero acceleration
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, len(actions) - 1)
        return self.greedy_action(x, y, v_x, v_y)

    def greedy_action(self, x, y, v_x, v_y):
        return self.policy[x][y][v_x][v_y]

    def start(self):
        return (0, random.choice(start_cols), 0, 0, CONTINUE)

    def random_start(self):
        x = random.randint(0, ROWS - 1)
        y = random.randint(0, COLS - 1)
        v_x = random.randint(0, MAX_VELOCITY)
        v_y = random.randint(0, MAX_VELOCITY)
        if self.out_of_bounds(x, y) or self.is_end(x, y):
            return self.random_start()
        return (x, y, v_x, v_y, CONTINUE)

    def random_action(self):
        return random.randint(0, len(actions) - 1)

    def next_state(self, x, y, v_x, v_y, action):
        a_x, a_y = actions[action]
        v_x = max(0, min(v_x + a_x, MAX_VELOCITY))
        v_y = max(0, min(v_y + a_y, MAX_VELOCITY))
        new_x = x + v_x
        new_y = y + v_y
        if new_y >= COLS - 1 and v_y != 0:
            # 检查是不是穿过了终点线
            time = (COLS - 1 - y) / v_y
            x_time = math.ceil(x + v_x * time)
            if self.is_end(x_time, COLS - 1):
                return (new_x, new_y, v_x, v_y, END, NOT_RESTART_ROUND)

        if self.out_of_bounds(new_x, new_y):
            return (*self.start(), RESTART_ROUND)
        return (new_x, new_y, v_x, v_y, CONTINUE, NOT_RESTART_ROUND)

    def out_of_bounds(self, x, y):
        if x < 0 or y < 0 or x >= ROWS or y >= COLS:
            return True
        return track[x][y] == 1

    def is_end(self, x, y):
        return (x, y) in fin_cells

    def generate_episode(self, not_es=True):
        episode = []
        # x = 0
        # y = random.choice(start_cols)
        # v_x = 0
        # v_y = 0
        # # action = self.action(x, y, v_x, v_y)
        # status = CONTINUE
        if not_es:
            x, y, v_x, v_y, status = self.start()
            action = self.action(x, y, v_x, v_y)
        else:
            x, y, v_x, v_y, status = self.random_start()
            action = self.random_action()
        while status == CONTINUE:
            episode.append((x, y, v_x, v_y, action))
            x, y, v_x, v_y, status, _ = self.next_state(x, y, v_x, v_y, action)
            if status == CONTINUE:
                action = self.action(x, y, v_x, v_y)
        return episode

    def iteration(self):
        for _ in range(100):
            for _ in range(10000):
                episode = self.generate_episode(not_es=False)
                episode = episode[::-1]
                G = 0
                for x, y, v_x, v_y, action in episode:
                    G -= 1
                    self.N[x][y][v_x][v_y][action] += 1
                    n = self.N[x][y][v_x][v_y][action]
                    q = self.Q[x][y][v_x][v_y][action]
                    self.Q[x][y][v_x][v_y][action] += (G - q) / n
                    self.policy[x][y][v_x][v_y] = np.argmax(self.Q[x][y][v_x][v_y])
            self.calculate_v_star()
            print(self.policy[0, 3:9, 0, 0])
            print(self.V[0, 3:9, 0, 0])
            # if (self.V[0, 3, 0, 0] > -15):
            #     self.print_greedy_track(0, 3, 0, 0)

    def value_iteration(self, theta=0.1):
        while True:
            delta = 0
            for x in range(ROWS):
                for y in range(COLS):
                    for v_x in range(MAX_VELOCITY + 1):
                        for v_y in range(MAX_VELOCITY + 1):
                            value = self.V[x][y][v_x][v_y]
                            max_value = MIN_VALUE
                            for a in range(ACTION_TYPE):
                                sum_val = 0
                                new_x, new_y, new_v_x, new_v_y, status, restart = self.next_state(x, y, v_x, v_y, a)
                                if restart == RESTART_ROUND:
                                    # 跳出去，否则状态函数不会趋于稳定，原因赞不知道
                                    continue
                                    # for s in start_cols:
                                    #     sum_val += 1/len(start_cols)*(-1 + self.V[0][s][0][0])
                                else:
                                    if status == CONTINUE:
                                        sum_val = -1 + self.V[new_x][new_y][new_v_x][new_v_y]
                                    elif status == END:
                                        sum_val = -1
                                max_value = max(max_value, sum_val)
                            self.V[x][y][v_x][v_y] = max_value
                            delta = max(delta, abs(max_value - value))
                            # if  0.1 < abs(max_value - value) and abs(max_value - value) < 0.2:
                            #     print()
            print(delta)
            if delta < theta:
                break

    def calculate_best_policy(self):
        for x in range(ROWS):
            for y in range(COLS):
                for v_x in range(MAX_VELOCITY + 1):
                    for v_y in range(MAX_VELOCITY + 1):
                        if self.out_of_bounds(x, y):
                            continue

                        # value = self.V[x][y][v_x][v_y]
                        max_value = MIN_VALUE
                        best_action = None
                        for a in range(ACTION_TYPE):
                            sum_val = 0
                            new_x, new_y, new_v_x, new_v_y, status, restart = self.next_state(x, y, v_x, v_y, a)
                            if restart == RESTART_ROUND:
                                continue
                                # for s in start_cols:
                                #     sum_val += 1 / len(start_cols) * (-1 + self.V[0][s][0][0])
                            else:
                                if status == CONTINUE:
                                    sum_val = -1 + self.V[new_x][new_y][new_v_x][new_v_y]
                                elif status == END:
                                    sum_val = -1
                            if max_value < sum_val:
                                max_value = sum_val
                                best_action = a
                        if best_action is None:
                            # 有些状态必定撞墙 比如(0,3,0,4)，无论怎么减速，必定撞到右面去
                            best_action = 0
                        self.policy[x][y][v_x][v_y] = best_action

    def calculate_v_star(self):
        for x in range(ROWS):
            for y in range(COLS):
                for v_x in range(MAX_VELOCITY + 1):
                    for v_y in range(MAX_VELOCITY + 1):
                        self.V[x][y][v_x][v_y] = max(self.Q[x][y][v_x][v_y])

    def print_greedy_track(self, x, y, v_x, v_y):
        status = CONTINUE
        while status == CONTINUE:
            action = self.greedy_action(x, y, v_x, v_y)
            print(x, y, v_x, v_y, action)
            x, y, v_x, v_y, status = self.next_state(x, y, v_x, v_y, action)



race = Racetrack()
# 蒙特卡洛最优V*[-10 -11 -11 -11 -11 -11],pi*=[1 1 1 1 1 1]
race.iteration()
race.calculate_v_star()

# DP获得的最优的V*[-10 -10 -10 -10 -10 -11]，pi*=[1 1 1 1 4 4]
# race.value_iteration()
# race.calculate_best_policy()
print(race.policy[0, 3:9, 0, 0])
print(race.V[0, 3:9, 0, 0])



