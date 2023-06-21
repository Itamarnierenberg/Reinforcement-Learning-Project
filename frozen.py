import time
from IPython.display import clear_output
import numpy as np
from FrozenLakeEnv import FrozenLakeEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
from Config import *
from CategoricalTD import categorical_td
from MonteCarlo import my_monte_carlo
from QLearning import q_learning
from matplotlib import cm


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


def categorial_mc(policy, x_axis, a=0.1, num_epochs=NUM_EPOCHS_MC, horizon=100, discount_factor=DISCOUNT_FACTOR):
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
                if i_star == len(x_axis) - 1:
                    break
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

def categorical_ql(locations, initial_prob, step_size=STEP_SIZE, num_epochs=NUM_EPOCHS_Q, discount_factor=DISCOUNT_FACTOR):
    q_table = np.zeros((26,4,X_AXIS_RESOLUTION))
    #each state and action define function
    for epoch in tqdm(range(num_epochs)):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        while not is_terminal:
            #finding action according to the maximal expectation
            action = find_action((q_table[curr_state],4)
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
                q_table[curr_state][action][i] = (1 - step_size) * q_table[curr_state][action][i] + step_size * p_list[i]
            curr_state = next_state
    env.reset()
    curr_state = env.get_state()
    action = action = find_action((q_table[curr_state],4)
    return q_table[curr_state][action]


def run_experiments_td_mc(init_state, num_experiments=10):
    our_policy = list()
    # for i in range(5):
    #     our_policy.append(RIGHT)
    for i in range(NUM_OF_STATES):
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
    y_axis_td = np.empty((num_experiments, X_AXIS_RESOLUTION))
    y_axis_mc = np.empty((num_experiments, X_AXIS_RESOLUTION))
    y_axis_mc_cat = np.empty((num_experiments, X_AXIS_RESOLUTION))
    for i in range(num_experiments):
        td_prob = np.array(categorical_td(env, our_policy, x_axis, init_prob, num_epochs=NUM_EPOCHS_TD))[init_state, :]
        monte_prob = my_monte_carlo(env, our_policy, x_axis, init_prob, num_epochs=NUM_EPOCHS_MC)
        mc_cat_prob = categorial_mc(our_policy, x_axis, discount_factor=DISCOUNT_FACTOR, num_epochs=NUM_EPOCHS_MC)
        fig.suptitle('TD Estimation and MC Estimation')
        # np.append(y_axis_td, np.array(td_prob), axis = 0)
        # np.append(y_axis_mc, np.array(monte_prob), axis = 0)
        # np.append(y_axis_mc_cat, np.array(mc_cat_prob), axis = 0)
        y_axis_td[i] = np.array(td_prob)
        y_axis_mc[i] = np.array(monte_prob)
        y_axis_mc_cat[i] = np.array(mc_cat_prob)
        print(f'TD Prob Sums To:{np.sum(y_axis_td[i])}')
        print(f'MC Prob Sums To:{np.sum(y_axis_mc[i])}')
        print(f'MC Categorial Prob Sums To:{np.sum(y_axis_mc_cat[i])}')
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


def run_experiments_q(init_state, num_experiments=10):
    x_axis = np.linspace(X_AXIS_LOWER_BOUND, X_AXIS_UPPER_BOUND, X_AXIS_RESOLUTION)
    y_axis_q = np.empty((num_experiments, X_AXIS_RESOLUTION))
    y_axis_3d_q = np.empty((num_experiments, NUM_ACTIONS, X_AXIS_RESOLUTION))
    y_axis_mc = np.empty((num_experiments, X_AXIS_RESOLUTION))
    for i in range(num_experiments):
        q_func = q_learning(env, x_axis, num_epochs=NUM_EPOCHS_Q)
        monte_prob = my_monte_carlo(env, x_axis, num_epochs=NUM_EPOCHS_MC)
        y_axis_q[i] = q_func[init_state, 0]
        y_axis_3d_q[i] = q_func[init_state]
        y_axis_mc[i] = np.array(monte_prob)
    y_axis_q_mean = np.mean(y_axis_q, axis=0)
    y_axis_3d_q_mean = np.mean(y_axis_3d_q, axis=0)
    y_axis_mc_mean = np.mean(y_axis_mc, axis=0)
    fig, axs = plt.subplots(2)
    axs[0].set_xlabel("Reward")
    axs[0].set_ylabel("QLearning Probability")
    axs[0].set_xlim(X_AXIS_LOWER_BOUND - 0.1, X_AXIS_UPPER_BOUND + 0.1)
    axs[0].set_ylim(0, np.max(y_axis_q_mean) + 0.02)
    axs[0].grid()
    axs[1].set_xlabel("Reward")
    axs[1].set_ylabel("MC Probability")
    axs[1].set_xlim(X_AXIS_LOWER_BOUND - 0.1, X_AXIS_UPPER_BOUND + 0.1)
    axs[0].set_ylim(0, np.max(y_axis_mc_mean) + 0.02)
    axs[1].grid()
    axs[0].plot(x_axis, y_axis_q_mean)
    axs[1].plot(x_axis, y_axis_mc_mean)
    # plt.xlabel("Reward")
    # plt.ylabel("Probability")
    # plt.xlim(X_AXIS_LOWER_BOUND - 0.1, X_AXIS_UPPER_BOUND + 0.1)
    # plt.ylim(0, np.max(y_axis_q_mean) + 0.02)
    # plt.grid()
    # plt.plot(x_axis, y_axis_q_mean)
    plt.show()
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # y, z = np.split(y_axis_3d_q_mean, 2)
    # print(len(y))
    # print(len(z))
    # surf = ax.plot_surface(x_axis, y, z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # plt.show()


if __name__ == "__main__":
    env = FrozenLakeEnv(MAPS[BOARD_SIZE])
    state = env.reset()
    print(state)
    print('Initial state:', state)
    print('Running TD and MC:')
    # run_experiments_td_mc(state, NUM_EXP)
    print('Running QLearning:')
    run_experiments_q(state, 10)



# Complicate the frozen lake, and when does it gets messy, how many trajectories do we need to make it work all of this is regarding the TD, MC should  work regardless
# Persistance -
# האם הספר מנסה להתחיל מנקודה כלשהי ולאפטם את כל המסלולים ביחס לאחוזון מסוים או שהספר אומר נגיע למצב מסוים ונדרוש שהחל מהמצב הזה ומהזמן הזה אני דורש משהו מהאחוזון הזה
# Optimization algorithms such as Q-Learning