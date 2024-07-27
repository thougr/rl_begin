import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BLACK_JACK_MAX_POINTS_PERMISSION = 21
DEALER_SHOW_STATE = 10
CARD_TYPE = 13
USABLE_ACE_STATE = 2
ACTION_TYPE = 2


class Blackjack(object):
    def __init__(self, eps=1.0):
        self.V = np.zeros((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, USABLE_ACE_STATE))
        self.Q = np.zeros((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, USABLE_ACE_STATE, ACTION_TYPE))
        self.N = np.zeros((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, USABLE_ACE_STATE, ACTION_TYPE))
        self.C = np.zeros((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, USABLE_ACE_STATE, ACTION_TYPE))
        self.policy = np.ones((DEALER_SHOW_STATE + 1, BLACK_JACK_MAX_POINTS_PERMISSION + 1, USABLE_ACE_STATE),
                              dtype=int)
        self.epsilon = eps
        self.setup()

    def setup(self):
        for dc in range(1, DEALER_SHOW_STATE + 1):
            for pc in range(20, BLACK_JACK_MAX_POINTS_PERMISSION + 1):
                for ace in range(USABLE_ACE_STATE):
                    self.policy[dc][pc][ace] = 0

    def card(self):
        return min(10, random.randint(1, CARD_TYPE))

    def draw_card(self, card_sum, usable_ace):
        new_card = self.card()
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
        return card_sum, usable_ace

    def bust(self, card_sum):
        return card_sum > 21

    # epsilon = 1, so randomly select
    def action(self, dealer_card, card_sum, usable_ace):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return self.policy[dealer_card][card_sum][usable_ace]

    def generate_episode(self):
        episode = []
        # dc_hidden = self.card()
        dealer_card = self.card()
        first_card = self.card()
        second_card = self.card()
        usable_ace = 1 if first_card == 1 or second_card == 1 else 0
        card_sum = first_card + second_card
        if usable_ace > 0:
            card_sum += 10
        action = self.action(dealer_card, card_sum, usable_ace)
        cur_state = (dealer_card, card_sum, usable_ace, action)
        episode.append(cur_state)
        while action == 1:
            card_sum, usable_ace = self.draw_card(card_sum, usable_ace)
            if self.bust(card_sum):
                return episode, -1
            else:
                action = self.action(dealer_card, card_sum, usable_ace)
                cur_state = (dealer_card, card_sum, usable_ace, action)
                episode.append(cur_state)
        dealer_sum = dealer_card
        dealer_usable_ace = 0
        if dealer_card == 1:
            dealer_sum += 10
            dealer_usable_ace = 1
        while dealer_sum < 17:
            dealer_sum, dealer_usable_ace = self.draw_card(dealer_sum, dealer_usable_ace)
            if self.bust(dealer_sum):
                return episode, 1
        if dealer_sum > card_sum:
            return episode, -1
        elif dealer_sum == card_sum:
            return episode, 0
        else:
            return episode, 1

    def iteration(self, use_first=True):
        for i in range(1000000):
            sequence, score = self.generate_episode()
            origin_len = len(sequence)
            if use_first:
                sequence = list(set(sequence))
                if len(sequence) != origin_len:
                    print("duplicate state")
            # reverse
            sequence = sequence[::-1]
            G = score
            W = 1
            for dealer_card, card_sum, usable_ace, action in sequence:
                self.C[dealer_card][card_sum][usable_ace][action] += W
                old_q = self.Q[dealer_card][card_sum][usable_ace][action]
                new_c = self.C[dealer_card][card_sum][usable_ace][action]
                self.Q[dealer_card][card_sum][usable_ace][action] += W / new_c * (G - old_q)
                old_action = self.policy[dealer_card][card_sum][usable_ace]
                other_action = 1 - old_action
                if self.Q[dealer_card][card_sum][usable_ace][other_action] > self.Q[dealer_card][card_sum][usable_ace][
                    old_action]:
                    self.policy[dealer_card][card_sum][usable_ace] = other_action
                if action != self.policy[dealer_card][card_sum][usable_ace]:
                    break
                W = W / 0.5
                # self.N[dealer_card][card_sum][usable_ace][action] += 1
                # n = self.N[dealer_card][card_sum][usable_ace][action]
                # v = self.Q[dealer_card][card_sum][usable_ace][action]
                # self.Q[dealer_card][card_sum][usable_ace][action] += (score - v) / n

    def calculate_value_star(self):
        for dc in range(1, DEALER_SHOW_STATE + 1):
            for pc in range(12, BLACK_JACK_MAX_POINTS_PERMISSION + 1):
                for ace in range(USABLE_ACE_STATE):
                    self.V[dc][pc][ace] = max(self.Q[dc][pc][ace][0], self.Q[dc][pc][ace][1])


def show_values(V, begin):
    data = V[1:, begin:]
    print(data.shape)
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(begin, data.shape[1] + begin), np.arange(1, data.shape[0] + 1))
    ax.plot_surface(x, y, data, cmap="viridis")
    # 设置轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # ax.view_init(azim=-150)
    plt.show()


bj = Blackjack()
# bj.iteration()
bj.iteration()
bj.calculate_value_star()
start = 12
print(bj.V[:, :, 0])
print(bj.V[:, :, 1])
no_usable_ace_data = bj.V[:, :, 0]
usable_ace_data = bj.V[:, :, 1]
show_values(no_usable_ace_data, start)
show_values(usable_ace_data, start)
print(bj.policy[:, :, 0])
print()
print(bj.policy[:, :, 1])
print(bj.Q[10,16,0,:], bj.N[10, 16, 0, :])
print(bj.Q[3, 12, 0, :], bj.N[3, 12, 0, :])
