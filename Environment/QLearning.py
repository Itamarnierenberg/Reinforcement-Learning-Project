import numpy as np
import Params as prm
from tqdm import tqdm
import random


def q_learning(env, x_axis, policy, step_size=prm.STEP_SIZE, discount_factor=prm.DISCOUNT_FACTOR, num_epochs=prm.NUM_EPOCHS_Q):
    num_states = len(policy)
    q_func = np.zeros((num_states, prm.NUM_ACTIONS, prm.X_AXIS_RESOLUTION))
    mc_prob = np.zeros((num_states, prm.NUM_ACTIONS, prm.X_AXIS_RESOLUTION))
    for state in tqdm(range(num_states), desc="Initializing Uniformly Distributed Q Function"):
        for action in range(prm.NUM_ACTIONS):
            for pos_reward in range(prm.X_AXIS_RESOLUTION):
                q_func[state][action][pos_reward] = 1/prm.X_AXIS_RESOLUTION
                mc_prob[state][action][pos_reward] = 1/prm.X_AXIS_RESOLUTION
    for epoch in tqdm(range(num_epochs), desc="QLearning:"):
        env.reset()
        curr_state = env.get_state()
        curr_state_idx = env.get_state_idx(curr_state)
        is_terminal = False
        first_iter = True
        while not is_terminal:
            action = policy(curr_state_idx)
            next_state, reward, is_terminal = env.step(action)
            next_state_idx = env.get_state_idx(next_state)
            p_list = np.zeros(prm.X_AXIS_RESOLUTION)
            # Projection
            for j in range(prm.X_AXIS_RESOLUTION):
                if is_terminal:
                    g = reward
                else:
                    g = reward + discount_factor * x_axis[j]
                if g <= x_axis[0]:
                    p_list[0] += q_func[next_state_idx, action, j]
                elif g >= x_axis[prm.X_AXIS_RESOLUTION - 1]:
                    p_list[prm.X_AXIS_RESOLUTION - 1] += q_func[next_state_idx, action, j]
                else:
                    i_star = 0
                    while x_axis[i_star + 1] <= g:
                        i_star += 1
                    eta = (g - x_axis[i_star]) / (x_axis[i_star + 1] - x_axis[i_star])
                    p_list[i_star] += (1 - eta) * q_func[next_state_idx, action, j]
                    p_list[i_star + 1] += eta * q_func[next_state_idx, action, j]
            # Incremental Step
            for j in range(prm.X_AXIS_RESOLUTION):
                q_func[curr_state, action, j] = (1 - step_size)*q_func[curr_state, action, j] + step_size*p_list[j]
            curr_state = next_state
    return q_func
