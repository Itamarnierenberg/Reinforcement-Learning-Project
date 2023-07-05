import matplotlib.pyplot as plt
import numpy as np
import Params as prm
from HazardEnv import HazardEnv
from CategoricalTD import categorical_td
from Utils import state_to_idx_dict
from Utils import create_control_data
from Utils import print_treatment_plan
from PolicyOptimization import policy_evaluation, policy_iteration
from rich.traceback import install
import random


install()


def run_td_exp(env, policy):
    x_axis = np.linspace(prm.X_AXIS_LOWER_BOUND, prm.X_AXIS_UPPER_BOUND, prm.X_AXIS_RESOLUTION)
    td_prob = categorical_td(env, policy, x_axis)
    print(td_prob[state_to_idx_dict(env.get_start_state(), state_list)])
    plt.plot(x_axis, td_prob[state_to_idx_dict(env.get_start_state(), env.get_state_space())])
    plt.show()


def run_policy_iteration(env, init_policy, treatment_prob=prm.TREATMENT_PROB, is_print = False):
    optimal_policy = policy_iteration(env, init_policy, treatment_prob =treatment_prob)
    if is_print:
        print_treatment_plan(env, state_list, optimal_policy)
    return optimal_policy


def run_big_experiment(init_policy, control_group, num_patients = 100, batch_num = 10):
    patients = []
    batch_size = int(num_patients/batch_num)
    for batch in range(batch_num):
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
                state_idx = state_to_idx_dict(patients[batch][patient].get_state())
                patients[batch][patient].step(current_policy[state_idx])
        #TODO update prob according to the old batches
        current_prob =


    return current_policy, current_prob


def run_regular_experiment (init_policy, control_group):
    env = HazardEnv(patient='Treatment', control_group=control_group)
    optimal_policy = run_policy_iteration(env, init_policy)
    return optimal_policy



if __name__ == '__main__':
    env = HazardEnv(patient='Control')
    init_policy = np.zeros(env.get_num_states())
    state_list = env.get_state_space()
    for state in state_list:
        init_policy[state_to_idx_dict(state, state_list)] = prm.TREATMENT_ACTION
    control_group = create_control_data()
    valuated_policy, valuated_prob = run_big_experiment(init_policy, control_group, num_patients = 1000, batch_num = 100)
    print_treatment_plan(env, valuated_policy, out_file= './aprox_policy.txt')
    optimal_policy = run_regular_experiment(init_policy, control_group)
    print_treatment_plan(env, optimal_policy, out_file='./optimal_policy.txt')
    # run_td_exp(env, valuated_policy)
    # run_td_exp(env, optimal_policy)









# if __name__ == '__main__':
#     control_group = create_control_data()
#     env = HazardEnv(patient='Treatment', control_group=control_group)
#     state_list = env.get_state_space()
#     num_states = env.get_num_states()
#     print(f'Num States = {num_states}')
#     init_policy = np.zeros(env.get_num_states())
#     for state in state_list:
#         init_policy[state_to_idx_dict(state, env.get_state_space())] = prm.TREATMENT_ACTION
#     optimal_policy = run_policy_iteration(env, init_policy, is_print=True)
#     run_td_exp(env, optimal_policy)
