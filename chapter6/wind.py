import random

import numpy as np

WIND_PUSH = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
ROWS = 7
COLS = 10
START = (3, 0)
GOAL = (3, 7)
ACTIONS = [
    (0, 1),  # 右
    (0, -1),  # 左
    (1, 0),  # 上
    (-1, 0),  # 下
]

ACTION_TYPE = len(ACTIONS)
MIN_VALUE = 0


class Wind:
    def __init__(self, eps=0.1, alpha=0.5):
        self.V = np.full(shape=(ROWS, COLS), fill_value=MIN_VALUE, dtype=np.float32)
        self.Q = np.full(shape=(ROWS, COLS, ACTION_TYPE), fill_value=MIN_VALUE, dtype=np.float32)
        self.policy = np.zeros(shape=(ROWS, COLS), dtype=np.uint8)
        self.epsilon = eps
        self.alpha = alpha

    def action(self, x, y):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_TYPE - 1)
        return self.greedy_action(x, y)

    def greedy_action(self, x, y):
        return self.policy[x][y]

    def next_state(self, x, y, action):
        dx, dy = ACTIONS[action]
        x += WIND_PUSH[y]
        x, y = x + dx, y + dy
        x = max(0, min(ROWS - 1, x))
        y = max(0, min(COLS - 1, y))
        return x, y

    def iteration(self, episodes=1000):
        step = 0
        for episode in range(episodes):
            x, y = START
            action = self.action(x, y)
            while (x, y) != GOAL:
                next_x, next_y = self.next_state(x, y, action)
                r = -1
                next_action = self.action(next_x, next_y)
                old_q = self.Q[x][y][action]
                self.Q[x][y][action] = old_q + self.alpha * (r + self.Q[next_x][next_y][next_action] - old_q)
                self.policy[x][y] = np.argmax(self.Q[x][y])
                x, y = next_x, next_y
                action = next_action
                # if step % 1000 == 0:
                #     print(episode, step)
                step += 1

            # self.calculate_v_star()
            # if self.V[START[0]][START[1]] > -18:
            #     print(f'Episode: {episode}, step: {len(self.trace())}')

    def calculate_v_star(self):
        for x in range(ROWS):
            for y in range(COLS):
                self.V[x][y] = max(self.Q[x][y])

    def trace(self):
        x, y = START
        route = []
        while (x, y) != GOAL:
            route.append((x, y))
            x, y = self.next_state(x, y, self.greedy_action(x, y))
        route.append(GOAL)
        return route


wind = Wind(alpha=0.1)
wind.iteration(episodes=1000)
wind.calculate_v_star()
print(wind.V)
print(wind.policy)
print(wind.V[START[0]][START[1]])
trace = wind.trace()
print(len(trace), trace)
