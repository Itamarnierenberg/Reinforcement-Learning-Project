import numpy as np
from Config import *
from tqdm import tqdm
from Policies import eps_greedy
import random


def q_learning(env, x_axis, step_size=STEP_SIZE, discount_factor=DISCOUNT_FACTOR, num_epochs=NUM_EPOCHS_Q):
    q_func = np.zeros((NUM_OF_STATES, NUM_ACTIONS, X_AXIS_RESOLUTION))
    mc_prob = np.zeros((NUM_OF_STATES, NUM_ACTIONS, X_AXIS_RESOLUTION))
    for state in tqdm(range(NUM_OF_STATES), desc="Initializing Uniformly Distributed Q Function"):
        for action in range(NUM_ACTIONS):
            for pos_reward in range(X_AXIS_RESOLUTION):
                q_func[state][action][pos_reward] = 1/X_AXIS_RESOLUTION
                mc_prob[state][action][pos_reward] = 1/X_AXIS_RESOLUTION
    for epoch in tqdm(range(num_epochs), desc="QLearning:"):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        first_iter = True
        while not is_terminal:
            next_action = eps_greedy(env, q_func, curr_state)
            if first_iter:
                action = 0
                first_iter = False
            else:
                action = next_action
            next_state, reward, is_terminal = env.step(action)
            p_list = np.zeros(X_AXIS_RESOLUTION)
            # Projection
            for j in range(X_AXIS_RESOLUTION):
                if is_terminal:
                    g = reward
                else:
                    g = reward + discount_factor * x_axis[j]
                if g <= x_axis[0]:
                    p_list[0] += q_func[next_state, action, j]
                elif g >= x_axis[X_AXIS_RESOLUTION - 1]:
                    p_list[X_AXIS_RESOLUTION - 1] += q_func[next_state, next_action, j]
                else:
                    i_star = 0
                    while x_axis[i_star + 1] <= g:
                        i_star += 1
                    eta = (g - x_axis[i_star]) / (x_axis[i_star + 1] - x_axis[i_star])
                    p_list[i_star] += (1 - eta) * q_func[next_state, next_action, j]
                    p_list[i_star + 1] += eta * q_func[next_state, next_action, j]
            # Incremental Step
            for j in range(X_AXIS_RESOLUTION):
                q_func[curr_state, action, j] = (1 - step_size)*q_func[curr_state, action, j] + step_size*p_list[j]
            curr_state = next_state
    return q_func
