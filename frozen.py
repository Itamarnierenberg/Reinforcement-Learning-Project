import time
from IPython.display import clear_output
import numpy as np
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

X_AXIS_LOWER_BOUND = -1
X_AXIS_UPPER_BOUND = 1
X_AXIS_RESOLUTION = 2
HORIZON = X_AXIS_RESOLUTION

MAPS = {
    "5x5": ["FFSFF",
            "FFFFF",
            "FFFFF",
            "FFFFF",
            "HFFFG"],
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


def categorical_td(policy, locations, initial_prob, step_size=0.1, num_epochs=500, discount_factor=1.0):
    return_prob = initial_prob
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
                    p_list[0] += return_prob[next_state][j]
                elif g >= locations[X_AXIS_RESOLUTION - 1]:
                    p_list[X_AXIS_RESOLUTION - 1] += return_prob[next_state][j]
                else:
                    i_star = 0
                    while locations[i_star + 1] <= g:
                        i_star += 1
                    eta = (g - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
                    #sif eta <=0:
                        #print(f'Eta = {eta}, g = {g}, location[i_star] = {locations[i_star]}, locations[i_star + 1] = {locations[i_star + 1]}')
                    p_list[i_star] += (1 - eta) * return_prob[next_state][j]
                    p_list[i_star + 1] += eta * return_prob[next_state][j]

            for i in range(X_AXIS_RESOLUTION):
                return_prob[curr_state][i] = (1 - step_size) * return_prob[curr_state][i] + step_size * p_list[i]
            curr_state = next_state
    return return_prob


our_policy = list()
# for i in range(5):
#     our_policy.append(RIGHT)
for i in range(0,25):
    our_policy.append(DOWN)
# print_solution(our_policy, env)
# Uniformly Distributed
init_prob = []
for i in range(len(our_policy)):
    init_prob.append([])
    for state in range(HORIZON):
        init_prob[i].append(1/HORIZON)

x_axis = np.linspace(X_AXIS_LOWER_BOUND, X_AXIS_UPPER_BOUND, X_AXIS_RESOLUTION)
td_prob = categorical_td(our_policy, x_axis, init_prob)
y_axis = np.array(td_prob)
plt.stem(x_axis, y_axis[2, :])
plt.xlabel("Reward")
plt.ylabel("Probability")
plt.xlim(-2, 2)
plt.ylim(0, 1)
plt.grid()
plt.show()
print(np.sum(y_axis[2,:]))
print(y_axis[2,:])