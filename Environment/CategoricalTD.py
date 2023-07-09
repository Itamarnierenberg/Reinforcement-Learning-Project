from tqdm import tqdm
import Params as prm
import numpy as np


def categorical_td(env, policy, locations, step_size=prm.STEP_SIZE, num_epochs=prm.NUM_EPOCHS_TD, discount_factor=prm.DISCOUNT_FACTOR):
    init_prob = []
    for i in range(len(policy)):
        init_prob.append([])
        for state in range(prm.X_AXIS_RESOLUTION):
            init_prob[i].append(1 / prm.X_AXIS_RESOLUTION)
    td_est_prob = init_prob
    for epoch in tqdm(range(num_epochs), desc='Categorical TD Epoch Progress'):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        step = 0
        while not is_terminal:
            curr_state_idx = env.get_state_idx(curr_state)
            action = policy[curr_state_idx]
            next_state, reward, is_terminal = env.step(action)
            p_list = np.zeros(prm.X_AXIS_RESOLUTION)
            next_state_idx = env.get_state_idx(next_state)
            for j in range(prm.X_AXIS_RESOLUTION):
                if is_terminal:
                    g = reward
                else:
                    g = reward + discount_factor * locations[j]
                if g <= locations[0]:
                    p_list[0] += td_est_prob[next_state_idx][j]
                elif g >= locations[prm.X_AXIS_RESOLUTION - 1]:
                    p_list[prm.X_AXIS_RESOLUTION - 1] += td_est_prob[next_state_idx][j]
                else:
                    i_star = 0
                    while locations[i_star + 1] <= g:
                        i_star += 1
                        if i_star == len(locations) - 1:
                            break
                    eta = (g - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
                    #sif eta <=0:
                        #print(f'Eta = {eta}, g = {g}, location[i_star] = {locations[i_star]}, locations[i_star + 1] = {locations[i_star + 1]}')
                    p_list[i_star] += (1 - eta) * td_est_prob[next_state_idx][j]
                    p_list[i_star + 1] += eta * td_est_prob[next_state_idx][j]

            for i in range(prm.X_AXIS_RESOLUTION):
                td_est_prob[curr_state_idx][i] = (1 - step_size) * td_est_prob[curr_state_idx][i] + step_size * p_list[i]
            curr_state = next_state
            step += 1
    return np.array(td_est_prob)
