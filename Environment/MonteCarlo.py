import numpy as np
from tqdm import tqdm
import Params as prm


def my_monte_carlo(env, locations, policy, num_epochs=prm.NUM_EPOCHS_MC, discount_factor=prm.DISCOUNT_FACTOR):
    est_prob = np.zeros(len(locations))
    return_counter = np.zeros(len(locations))
    for epoch in tqdm(range(num_epochs)):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        iter = 0
        traj_return = 0
        state_idx = env.get_state_idx(curr_state)
        while not is_terminal:
            action = policy[state_idx]
            next_state, reward, is_terminal = env.step(action)
            traj_return += (discount_factor ** iter) * reward
            iter += 1
            state_idx = env.get_state_idx(next_state)
        i_star = 0
        while locations[i_star + 1] <= traj_return:
            i_star += 1
            if i_star == len(locations) - 1:
                break
        return_counter[i_star] += 1
        est_prob = return_counter / (epoch + 1)

    return est_prob