import numpy as np
from tqdm import tqdm
from Config import *


def my_monte_carlo(env, policy, locations, initial_prob, num_epochs=NUM_EPOCHS_MC, discount_factor=DISCOUNT_FACTOR):
    est_prob = np.zeros(len(locations))
    return_counter = np.zeros(len(locations))
    for epoch in tqdm(range(num_epochs)):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        iter = 0
        traj_return = 0
        while not is_terminal:
            action = policy[curr_state]
            next_state, reward, is_terminal = env.step(action)
            traj_return += (discount_factor ** iter) * reward
            iter += 1
        i_star = 0
        while locations[i_star + 1] <= traj_return:
            i_star += 1
            if i_star == len(locations) - 1:
                break
        eta = (traj_return - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
        return_counter[i_star] += 1
        est_prob = return_counter / (epoch + 1)
        #est_prob[i_star] = (return_counter[i_star] * (1 - eta)) / (epoch + 1)
        #est_prob[i_star + 1] = (return_counter[i_star] * eta) / (epoch + 1)

    return est_prob