import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BLACK_JACK_MAX_POINTS_PERMISSION = 21
DEALER_SHOW_STATE = 10
CARD_TYPE = 13


class Blackjack(object):
    def __init__(self):
        self.V = np.zeros((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, 2))
        self.N = np.zeros((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, 2))
        self.policy = np.ones((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, 2))
        self.setup()

    def setup(self):
        for dc in range(1, DEALER_SHOW_STATE + 1):
            for pc in range(20, BLACK_JACK_MAX_POINTS_PERMISSION + 1):
                for ace in range(2):
                    self.policy[dc][pc][ace] = 0

    def card(self):
        return min(10, random.randint(1, CARD_TYPE))

    def draw_card(self, card_sum, usable_ace):
        card = self.card()
        card_sum += card


    def generate_episode(self):
        episode = []
        dc_hidden = self.card()
        dealer_card = self.card()
        first_card = self.card()
        second_card = self.card()
        usable_ace = 1 if first_card == 1 or second_card == 1 else 0
        card_sum = first_card + second_card
        if usable_ace > 0:
            card_sum += 10
        cur_state = (dealer_card, card_sum, usable_ace)
        episode.append(cur_state)
        while card_sum < 20:
            new_card = min(np.random.randint(CARD_TYPE) + 1, 10)
            if new_card > 1:
                card_sum += new_card
                if card_sum > 21 and usable_ace:
                    card_sum -= 10
                    usable_ace = 0
            elif new_card == 1:
                if card_sum <= 10:
                    card_sum += 11
                    usable_ace = 1
                else:
                    card_sum += 1
            if card_sum > 21:
                return episode, -1
            else:
                cur_state = (dealer_card, card_sum, usable_ace)
                episode.append(cur_state)
        dealer_sum = dealer_card
        dealer_usable_ace = 0
        if dealer_card == 1:
            dealer_sum += 10
            dealer_usable_ace = 1
        while dealer_sum < 17:
            new_card = min(np.random.randint(CARD_TYPE) + 1, 10)
            if new_card > 1:
                dealer_sum += new_card
                if dealer_sum > 21 and dealer_usable_ace:
                    dealer_sum -= 10
                    dealer_usable_ace = 0
                elif new_card == 1:
                    if dealer_sum <= 10:
                        dealer_sum += 11
                        dealer_usable_ace = 1
                    else:
                        dealer_sum += 1
            if dealer_sum > 21:
                return episode, 1
        if dealer_sum > card_sum:
            return episode, -1
        elif dealer_sum == card_sum:
            return episode, 0
        else:
            return episode, 1

    def iteration(self):
        for i in range(500000):
            sequence, score = self.generate_episode()
            sequence = list(set(sequence))
            for _, seq in enumerate(sequence):
                self.N[seq[0]][seq[1]][seq[2]] += 1
                self.V[seq[0]][seq[1]][seq[2]] += 1/self.N[seq[0]][seq[1]][seq[2]]*(score -self.V[seq[0]][seq[1]][seq[2]])


bj = Blackjack()
bj.iteration()
begin = 4
no_usable_ace_data = bj.V[1:,begin:,0]
usable_ace_data = bj.V[1:,begin:,1]
print(no_usable_ace_data)
print(usable_ace_data)
figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')
print(no_usable_ace_data.shape, no_usable_ace_data.shape[0], no_usable_ace_data.shape[1])
x, y = np.meshgrid(np.arange(begin, no_usable_ace_data.shape[1] + begin), np.arange(1, no_usable_ace_data.shape[0] + 1))
ax.plot_surface(x, y, no_usable_ace_data, cmap="viridis")
ax.view_init(azim=200)

# 设置轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()


