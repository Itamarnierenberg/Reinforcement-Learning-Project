from tqdm import tqdm
from Config import *
import numpy as np


def categorical_td(env, policy, locations, initial_prob, step_size=STEP_SIZE, num_epochs=NUM_EPOCHS_TD, discount_factor=DISCOUNT_FACTOR):
    td_est_prob = initial_prob
    for epoch in tqdm(range(num_epochs)):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        while not is_terminal:
            action = policy[curr_state]
            next_state, reward, is_terminal = env.step(action)
            p_list = np.zeros(X_AXIS_RESOLUTION)
            for j in range(X_AXIS_RESOLUTION):
                if is_terminal:
                    g = reward
                else:
                    g = reward + discount_factor * locations[j]
                if g <= locations[0]:
                    p_list[0] += td_est_prob[next_state][j]
                elif g >= locations[X_AXIS_RESOLUTION - 1]:
                    p_list[X_AXIS_RESOLUTION - 1] += td_est_prob[next_state][j]
                else:
                    i_star = 0
                    while locations[i_star + 1] <= g:
                        i_star += 1
                        if i_star == len(locations) - 1:
                            break
                    eta = (g - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
                    #sif eta <=0:
                        #print(f'Eta = {eta}, g = {g}, location[i_star] = {locations[i_star]}, locations[i_star + 1] = {locations[i_star + 1]}')
                    p_list[i_star] += (1 - eta) * td_est_prob[next_state][j]
                    p_list[i_star + 1] += eta * td_est_prob[next_state][j]

            for i in range(X_AXIS_RESOLUTION):
                td_est_prob[curr_state][i] = (1 - step_size) * td_est_prob[curr_state][i] + step_size * p_list[i]
            curr_state = next_state
    return td_est_prob