import time
from IPython.display import clear_output
import numpy as np
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

X_AXIS_LOWER_BOUND = -1
X_AXIS_UPPER_BOUND = 1
X_AXIS_RESOLUTION = 100
HORIZON = X_AXIS_RESOLUTION
GRAPH_TYPE = 'plot'

MAPS = {
    "5x5": ["FFSFF",
            "FFFFF",
            "FFFFF",
            "FFFFF",
            "FFFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFHFFHFL",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHTFHFFL",
        "FFFHFFFG",
    ],
}

env = FrozenLakeEnv(MAPS["5x5"])
state = env.reset()
print(state)
print('Initial state:', state)


class RandomAgent():
    def __init__(self):
        self.env = None

    def animation(self, epochs: int, state: int, action: List[int], total_cost: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Total Cost: {total_cost}")
        time.sleep(1)

    def random_search(self, FrozenLakeEnv: env) -> Tuple[List[int], int]:
        self.env = env
        self.env.reset()
        epochs = 0
        cost = 0
        total_cost = 0

        actions = []

        state = self.env.get_initial_state()
        while not self.env.is_final_state(state):
            action = self.env.action_space.sample()
            new_state, cost, terminated = self.env.step(action)

            while terminated is True and self.env.is_final_state(state) is False:
                self.env.set_state(state)
                action = self.env.action_space.sample()
                new_state, cost, terminated = self.env.step(action)

            actions.append(action)
            total_cost += cost
            state = new_state
            epochs += 1

            self.animation(epochs, state, action, total_cost)

        return (actions, total_cost)


def print_solution(actions, env: FrozenLakeEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
        state, cost, terminated = env.step(action)
        total_cost += cost
        clear_output(wait=True)

        print(env.render())
        print(f"Timestep: {i + 2}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Cost: {cost}")
        print(f"Total cost: {total_cost}")

        time.sleep(1)

        if terminated is True:
            break


def my_monte_carlo(policy, locations, initial_prob, num_epochs=500000, discount_factor=0.99):
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
        eta = (traj_return - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
        return_counter[i_star] += 1
        est_prob = return_counter / (epoch + 1)
        #est_prob[i_star] = (return_counter[i_star] * (1 - eta)) / (epoch + 1)
        #est_prob[i_star + 1] = (return_counter[i_star] * eta) / (epoch + 1)

    return est_prob


def categorical_td(policy, locations, initial_prob, step_size=0.1, num_epochs=500, discount_factor=0.99):
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
                    eta = (g - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
                    #sif eta <=0:
                        #print(f'Eta = {eta}, g = {g}, location[i_star] = {locations[i_star]}, locations[i_star + 1] = {locations[i_star + 1]}')
                    p_list[i_star] += (1 - eta) * td_est_prob[next_state][j]
                    p_list[i_star + 1] += eta * td_est_prob[next_state][j]

            for i in range(X_AXIS_RESOLUTION):
                td_est_prob[curr_state][i] = (1 - step_size) * td_est_prob[curr_state][i] + step_size * p_list[i]
            curr_state = next_state
    return td_est_prob


def categorial_mc(policy, x_axis, a=0.1, num_epochs=500000, horizon=100, discount_factor=1.0):
    p = np.full(X_AXIS_RESOLUTION, 1/X_AXIS_RESOLUTION)
    for epoch in tqdm(range(num_epochs)):
        t = 0
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        trajectory = []
        gama = 1
        while not is_terminal:
            action = policy[curr_state]
            next_state, reward, is_terminal = env.step(action)
            trajectory.append((curr_state,gama * reward))
            curr_state = next_state
            gama = gama * discount_factor
            t = t+1
        g = 0
        for i in range(t-1, -1, -1):
            curr_reward = trajectory[i][1]
            curr_state = trajectory[i][0]
            g = g + curr_reward
            # if i < t - horizon:
            #     g = g - trajectory[i+horizon][1]
            i_star = 0
            while x_axis[i_star + 1] <= g:
                i_star += 1
            if curr_state not in [t[0] for t in trajectory[:i]] and curr_state == 2:
                for j in range(X_AXIS_RESOLUTION):
                    if j == i_star :
                        p[j] = (1-a)*p[j] + a
                    else:
                        p[j] = (1 - a) * p[j]
    return p

def calculate_expectation (prob):
    exp = 0
    delta = (X_AXIS_UPPER_BOUND - X_AXIS_LOWER_BOUND) / X_AXIS_RESOLUTION
    for i in range(X_AXIS_RESOLUTION):
        exp += (1 - prob[i])*delta

def find_action (state_ql_table, actions_num):
    actions = []
    max=0
    for i in range (actions_num):
        curr = calculate_expectation(ql_table[i])
        if curr > max:
            actions.clear()
            actions.append(i)
            max = curr
        elif curr ==  max:
            actions.append(i)

# def categorical_ql(locations, initial_prob, step_size=0.1, num_epochs=500, discount_factor=0.99):
#     q_table = np.zeros((26,4,X_AXIS_RESOLUTION))
#     #each state and action define function
#     for epoch in tqdm(range(num_epochs)):
#         env.reset()
#         curr_state = env.get_state()
#         is_terminal = False
#         while not is_terminal:
#             #finding action according to the maximal expectation
#             action = find_action((q_table[curr_state],4)
#             next_state, reward, is_terminal = env.step(action)
#             p_list = np.zeros(X_AXIS_RESOLUTION)
#             for j in range(X_AXIS_RESOLUTION):
#                 if is_terminal:
#                     g = reward
#                 else:
#                     g = reward + discount_factor * locations[j]
#                 if g <= locations[0]:
#                     p_list[0] += td_est_prob[next_state][j]
#                 elif g >= locations[X_AXIS_RESOLUTION - 1]:
#                     p_list[X_AXIS_RESOLUTION - 1] += td_est_prob[next_state][j]
#                 else:
#                     i_star = 0
#                     while locations[i_star + 1] <= g:
#                         i_star += 1
#                     eta = (g - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
#                     #sif eta <=0:
#                         #print(f'Eta = {eta}, g = {g}, location[i_star] = {locations[i_star]}, locations[i_star + 1] = {locations[i_star + 1]}')
#                     p_list[i_star] += (1 - eta) * td_est_prob[next_state][j]
#                     p_list[i_star + 1] += eta * td_est_prob[next_state][j]
#
#             for i in range(X_AXIS_RESOLUTION):
#                 q_table[curr_state][action][i] = (1 - step_size) * q_table[curr_state][action][i] + step_size * p_list[i]
#             curr_state = next_state
#     env.reset()
#     curr_state = env.get_state()
#     action = action = find_action((q_table[curr_state],4)
#     return q_table[curr_state][action]


def run_experiments (num_experiments = 10):
    our_policy = list()
    # for i in range(5):
    #     our_policy.append(RIGHT)
    for i in range(0, 25):
        our_policy.append(DOWN)
    # print_solution(our_policy, env)
    # Uniformly Distributed
    init_prob = []
    for i in range(len(our_policy)):
        init_prob.append([])
        for state in range(HORIZON):
            init_prob[i].append(1/HORIZON)

    x_axis = np.linspace(X_AXIS_LOWER_BOUND, X_AXIS_UPPER_BOUND, X_AXIS_RESOLUTION)
    fig, axs = plt.subplots(3)
    y_axis_td = np.empty((num_experiments,X_AXIS_RESOLUTION))
    y_axis_mc = np.empty((num_experiments,X_AXIS_RESOLUTION))
    y_axis_mc_cat = np.empty((num_experiments, X_AXIS_RESOLUTION))
    for i in range(num_experiments):
        td_prob = np.array(categorical_td(our_policy, x_axis, init_prob))[2,:]
        monte_prob = my_monte_carlo(our_policy, x_axis, init_prob)
        mc_cat_prob = categorial_mc(our_policy, x_axis, discount_factor=0.99)
        fig.suptitle('TD Estimation and MC Estimation')
        # np.append(y_axis_td, np.array(td_prob), axis = 0)
        # np.append(y_axis_mc, np.array(monte_prob), axis = 0)
        # np.append(y_axis_mc_cat, np.array(mc_cat_prob), axis = 0)
        y_axis_td[i] = np.array(td_prob)
        y_axis_mc[i] = np.array(monte_prob)
        y_axis_mc_cat[i] = np.array(mc_cat_prob)
    y_axis_td_mean = np.mean(y_axis_td, axis=0)
    y_axis_mc_mean = np.mean(y_axis_mc, axis=0)
    y_axis_mc_cat_mean = np.mean(y_axis_mc_cat, axis=0)
    y_axis_td_max = np.max(y_axis_td, axis=0)
    y_axis_mc_max = np.max(y_axis_mc, axis=0)
    y_axis_mc_cat_max = np.max(y_axis_mc_cat, axis=0)
    y_axis_td_min = np.min(y_axis_td, axis=0)
    y_axis_mc_min = np.min(y_axis_mc, axis=0)
    y_axis_mc_cat_min = np.min(y_axis_mc_cat, axis=0)
    axs[0].set_xlabel("Reward")
    axs[0].set_ylabel("TD Probability")
    axs[0].set_xlim(X_AXIS_LOWER_BOUND - 0.1, X_AXIS_UPPER_BOUND + 0.1)
    axs[0].set_ylim(0, max(np.max(y_axis_td_max), np.max(y_axis_mc_max), np.max(y_axis_mc_cat_max)) + 0.02)
    axs[0].grid()
    axs[1].set_xlabel("Reward")
    axs[1].set_ylabel("MC Probability")
    axs[1].set_xlim(X_AXIS_LOWER_BOUND - 0.1, X_AXIS_UPPER_BOUND + 0.1)
    axs[1].set_ylim(0, max(np.max(y_axis_td_max), np.max(y_axis_mc_max), np.max(y_axis_mc_cat_max)) + 0.02)
    axs[1].grid()
    axs[2].set_xlabel("Reward")
    axs[2].set_ylabel("Categorial MC Probability")
    axs[2].set_xlim(X_AXIS_LOWER_BOUND - 0.1, X_AXIS_UPPER_BOUND + 0.1)
    axs[2].set_ylim(0, max(np.max(y_axis_td), np.max(y_axis_mc), np.max(y_axis_mc_cat)) + 0.02)
    axs[2].grid()
    if GRAPH_TYPE == 'stem':
        axs[0].stem(x_axis, y_axis_td_mean)
        axs[0].stem(x_axis, y_axis_td_max, linestyle='dashed')
        axs[0].stem(x_axis, y_axis_td_min, linestyle='dashed')
        axs[1].stem(x_axis, y_axis_mc_mean)
        axs[1].stem(x_axis, y_axis_mc_max, linestyle='dashed')
        axs[1].stem(x_axis, y_axis_mc_min, linestyle='dashed')
        axs[2].stem(x_axis, y_axis_mc_cat_mean)
        axs[2].stem(x_axis, y_axis_mc_cat_max, linestyle='dashed')
        axs[2].stem(x_axis, y_axis_mc_cat_min, linestyle='dashed')
    if GRAPH_TYPE == 'plot':
        axs[0].plot(x_axis, y_axis_td_mean)
        axs[0].plot(x_axis, y_axis_td_max, linestyle='dashed')
        axs[0].plot(x_axis, y_axis_td_min, linestyle='dashed')
        axs[1].plot(x_axis, y_axis_mc_mean)
        axs[1].plot(x_axis, y_axis_mc_max, linestyle='dashed')
        axs[1].plot(x_axis, y_axis_mc_min, linestyle='dashed')
        axs[2].plot(x_axis, y_axis_mc_cat_mean)
        axs[2].plot(x_axis, y_axis_mc_cat_max, linestyle='dashed')
        axs[2].plot(x_axis, y_axis_mc_cat_min, linestyle='dashed')
    plt.show()

if __name__ == "__main__":
    run_experiments(2)

    # print(f'TD Prob Sums To:{np.sum(y_axis_td[2, :])}')
    # print(f'MC Prob Sums To:{np.sum(y_axis_mc)}')
    # print(f'MC Categorial Prob Sums To:{np.sum(y_axis_mc_cat)}')

# Complicate the frozen lake, and when does it gets messy, how many trajectories do we need to make it work all of this is regarding the TD, MC should  work regardless
# Persistance -
# האם הספר מנסה להתחיל מנקודה כלשהי ולאפטם את כל המסלולים ביחס לאחוזון מסוים או שהספר אומר נגיע למצב מסוים ונדרוש שהחל מהמצב הזה ומהזמן הזה אני דורש משהו מהאחוזון הזה
# Optimization algorithms such as Q-Learning