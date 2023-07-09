import matplotlib.pyplot as plt
import numpy as np
import Params as prm
from HazardEnv import HazardEnv
from CategoricalTD import categorical_td
from MonteCarlo import my_monte_carlo
from Utils import create_control_data
from Utils import print_treatment_plan
from Utils import print_learned_dist
from PolicyOptimization import policy_evaluation, policy_iteration, perform_interactions
from rich.traceback import install
import random
from tqdm import tqdm
from datetime import datetime
import os


install()


def run_dist_exp(env, policy):
    x_axis = np.linspace(prm.X_AXIS_LOWER_BOUND, prm.X_AXIS_UPPER_BOUND, prm.X_AXIS_RESOLUTION)
    td_prob = categorical_td(env, policy)
    mc_prob = my_monte_carlo(env, x_axis, policy)
    state_idx = env.get_state_idx(env.get_start_state())
    print(f'X Axis = {x_axis}')
    print(f'TD Prob = {td_prob[state_idx]}')
    print(f'MC Prob = {mc_prob}')
    print(f'TD Sum = {np.sum(td_prob[state_idx])}')
    print(f'MC Sum = {np.sum(mc_prob)}')
    print(td_prob[env.get_state_idx(env.get_start_state())])
    folder_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    curr_dir = f'./results/{folder_name}'
    os.mkdir(curr_dir)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(x_axis, td_prob[state_idx], label='td_prob')
    plt.plot(x_axis, mc_prob, label='mc_prob')
    plt.ylabel('Probability')
    plt.xlabel("Reward")
    plt.title('TD/MC Reward Distribution Estimation')
    plt.legend(loc='upper right')
    plt.savefig(f'{curr_dir}/{prm.TD_MC_PLOT}')
    plt.show()


def run_policy_iteration(env, init_policy, treatment_prob=prm.TREATMENT_PROB, is_print=False):
    optimal_policy = policy_iteration(env, init_policy, treatment_prob=treatment_prob)
    if is_print:
        print_treatment_plan(env, optimal_policy)
    return optimal_policy


def run_big_experiment(init_policy, control_group, num_patients = 100, batch_num = 10):
    patients = []
    batch_size = int(num_patients/batch_num)
    for batch in tqdm(range(batch_num)):
        patients.append([])
        for j in range(batch_size):
            patients[batch].append(HazardEnv(patient='Treatment', control_group=control_group))
    num_states = patients[0][0].get_num_states()
    state_list = patients[0][0].get_state_space()
    current_policy = init_policy
    current_prob = [1/3, 1/3, 1/3]

    # start the algorithm
    for batch in range(batch_num):
        print("evaluate on batch num:")
        print(batch)
        chosen_patient = random.randint(0, batch_size-1)
        print("chosen patient is:")
        print(chosen_patient)
        current_policy = run_policy_iteration(patients[batch][chosen_patient], current_policy, treatment_prob = current_prob)
        #update the next steps for all the people in the experiment until the moment
        #assume same probability for all the steps, so we can run the batch one step only and watch the data
        #we evaluate the new prob for all the state together
        #currently implemented only for one feature, so there is only one prob vector
        down = 0
        up = 0
        stay = 0
        total = 0
        for i in range(batch + 1):
            for patient in range(batch_size):
                if not patients[batch][patient].is_terminal_state():
                    current_state = patients[batch][patient].get_state()
                    curr_val = current_state[0]
                    state_idx = state_to_idx_dict(current_state, state_list)
                    patients[batch][patient].step(current_policy[state_idx])
                    new_val = patients[batch][patient].get_state()[0]
                    if new_val > curr_val:
                        up = up + 1
                    elif new_val < curr_val:
                        down = down + 1
                    else:
                        stay = stay + 1
                    total = total + 1
        current_prob = [down/total, stay/total, up/total]

    return current_policy, current_prob


def run_regular_experiment (init_policy, control_group):
    env = HazardEnv(patient='Treatment', control_group=control_group)
    optimal_policy = run_policy_iteration(env, init_policy)
    return optimal_policy


# if __name__ == '__main__':
#     env = HazardEnv(patient='Control')
#     init_policy = np.zeros(env.get_num_states())
#     state_list = env.get_state_space()
#     for state in state_list:
#         init_policy[state_to_idx_dict(state, state_list)] = prm.TREATMENT_ACTION
#     control_group = create_control_data()
#     valuated_policy, valuated_prob = run_big_experiment(init_policy, control_group, num_patients = 100, batch_num = 10)
#     print('koko')
#     print_treatment_plan(env, valuated_policy, out_file= './aprox_policy.txt')
#     optimal_policy = run_regular_experiment(init_policy, control_group)
#     print_treatment_plan(env, optimal_policy, out_file='./optimal_policy.txt')
#     # run_td_exp(env, valuated_policy)
#     # run_td_exp(env, optimal_policy)

if __name__ == '__main__':
    control_group = create_control_data()
    env = HazardEnv(patient='Treatment', control_group=control_group)
    state_list = env.get_state_space()
    num_states = env.get_num_states()
    print(f'Num States = {num_states}')
    init_policy = np.zeros(env.get_num_states())
    for state in state_list:
        init_policy[env.get_state_idx(state)] = prm.TREATMENT_ACTION
    for i in range(prm.SIZE_OF_ACTION_GROUP):
        env.reset()
        while not env.is_terminal:
            perform_interactions(env, init_policy)
    optimal_policy = run_policy_iteration(env, init_policy, is_print=True)
    if prm.BAYES_MODE:
        print_learned_dist(env)
    print("done")
    # run_td_exp(env, optimal_policy)
    run_dist_exp(env, optimal_policy)
