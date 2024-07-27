import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 全局变量
Q = None
policy = None
N = None
dc = None
pc = None
ace = None
episode = None

def card():
    return min(10, random.randint(1, 13))

def setup():
    global Q, N, policy
    Q = np.zeros((11, 22, 2, 2))
    N = np.zeros((11, 22, 2, 2))
    policy = np.ones((11, 22, 2), dtype=int)
    for dc in range(1, 11):
        for pc in range(20, 22):
            for ace in range(2):
                policy[dc, pc, ace] = 0

def exploring_episode():
    global dc, pc, ace, episode
    episode = []
    dc_hidden = card()
    dc = random.randint(1, 10)
    ace = bool(random.randint(0, 1))
    pc = 12 + random.randint(0, 9)
    action = random.randint(0, 1)
    while True:
        episode.append((dc, pc, ace, action))
        if action == 1:
            draw_card()
            if bust():
                break
            action = policy[dc, pc, int(ace)]
        else:
            break
    outcome_value = outcome(dc, dc_hidden)
    learn(episode, outcome_value)
    return outcome_value, episode

def episode():
    global dc, pc, ace, episode
    episode = []
    dc_hidden = card()
    dc = card()
    pcard1 = card()
    pcard2 = card()
    ace = pcard1 == 1 or pcard2 == 1
    pc = pcard1 + pcard2
    if ace:
        pc += 10
    if pc != 21:  # natural blackjack ends all
        while True:
            episode.append((dc, pc, ace))
            if policy[dc, pc, int(ace)] == 1:
                draw_card()
                if bust():
                    break
            else:
                break
    outcome_value = outcome(dc, dc_hidden)
    learn(episode, outcome_value)
    return outcome_value, episode

def learn(episode, outcome):
    for dc, pc, ace, action in episode:
        if pc > 11:
            N[dc, pc, int(ace), action] += 1
            Q[dc, pc, int(ace), action] += (outcome - Q[dc, pc, int(ace), action]) / N[dc, pc, int(ace), action]
            policy_action = policy[dc, pc, int(ace)]
            other_action = 1 - policy_action
            if Q[dc, pc, int(ace), other_action] > Q[dc, pc, int(ace), policy_action]:
                policy[dc, pc, int(ace)] = other_action

def outcome(dc, dc_hidden):
    dace = dc == 1 or dc_hidden == 1
    dcount = dc + dc_hidden
    if dace:
        dcount += 10
    dnatural = (dcount == 21)
    pnatural = (len(episode) == 0)
    if pnatural and dnatural:
        return 0
    elif pnatural:
        return 1
    elif dnatural:
        return -1
    elif bust():
        return -1
    else:
        while dcount < 17:
            card_value = card()
            dcount += card_value
            if not dace and card_value == 1:
                dcount += 10
                dace = 1
            if dace and dcount > 21:
                dcount -= 10
                dace = 0
        if dcount > 21:
            return 1
        elif dcount > pc:
            return -1
        elif dcount == pc:
            return 0
        else:
            return 1

def draw_card():
    global pc, ace
    card_value = card()
    pc += card_value
    if not ace and card_value == 1:
        pc += 10
        ace = 1
    if ace and pc > 21:
        pc -= 10
        ace = 0

def bust():
    return pc > 21

def gr(source, ace, action, arr=None):
    if arr is None:
        arr = np.zeros((10, 10))
    ace = int(ace)
    for i in range(10):
        for j in range(10):
            arr[i, j] = source[i+1, j+12, ace, action]
    return arr

def grp(ace, arr=None):
    if arr is None:
        arr = np.zeros((10, 10))
    ace = int(ace)
    for i in range(10):
        for j in range(10):
            arr[i, j] = policy[i+1, j+12, ace]
    return arr

def plot_surface(arr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(arr.shape[0])
    y = np.arange(arr.shape[1])
    x, y = np.meshgrid(x, y)
    z = arr
    ax.plot_surface(x, y, z, cmap='viridis')
    plt.show()

def experiment():
    setup()
    for count in range(500):
        for _ in range(1000):
            exploring_episode()
    ar0 = np.zeros((10, 10))
    ar1 = np.zeros((10, 10))
    print(count)
    ar0 = gr(Q, False, 0, ar0)
    ar1 = gr(Q, True, 1, ar1)
    plot_surface(ar0)
    plot_surface(ar1)

# 运行实验
experiment()
print(policy[:,:,0])
print()
print(policy[:,:,1])
