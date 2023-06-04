from Config import *
import random
import numpy as np
import math


def eps_greedy(q_func, state, epsilon=0.5):
    p = random.random()
    action = None
    if p < epsilon:
        action = random.choice(ACTION_LIST)
    else:
        max_exp = -math.inf
        for curr_action in ACTION_LIST:
            curr_exp = np.mean(q_func[state][curr_action])
            if curr_exp > max_exp:
                action = curr_action
                max_exp = curr_exp
    return action


