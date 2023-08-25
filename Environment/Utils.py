import Params as prm
import numpy as np
from tqdm import tqdm
from HazardEnv import HazardEnv
from rich.traceback import install
from datetime import datetime
import os
import matplotlib.pyplot as plt
install()


def print_control_data(control_group):
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        for feature in prm.FEATURES:
            print(f'Patient Number {i} Body Temp = {control_group[i][feature["name"]]}')


def create_control_data():
    control_group = list()
    for patient in tqdm(range(prm.SIZE_OF_CONTROL_GROUP), desc='Simulating Control Group'):
        curr_env = HazardEnv(patient='Control')
        while not curr_env.is_terminal:
            curr_env.step(prm.CONTROL_ACTION)
        control_group.append(curr_env.get_history())
    return control_group


def print_treatment_plan(env, policy, out_file=prm.POLICY_OUTPUT_FILE):
    policy_str = ''
    state_list = env.get_state_space()
    for state_idx, action in enumerate(policy):
        if env.is_terminal_state(state_list[state_idx]):
            continue
        state = state_list[state_idx]
        policy_str += f'Under the Measurements:\n'
        for idx, feature_value in enumerate(state):
            policy_str += f'\t\t{prm.FEATURES_IDX_TO_NAME[idx]} = {feature_value}\n'
        policy_str += f'{prm.ACTION_IDX_TO_NAME[int(action)]}\n\n'
    with open(out_file, prm.WRITE_FILE) as my_file:
        my_file.write(policy_str)


def find_reachable_states(env):
    reachable_states = set()
    add_state(env, env.get_start_state(), reachable_states)
    return reachable_states


def print_learned_dist(env, out_file=prm.PROB_OUPUT_FILE):
    prob_str = ''
    state_list = env.get_state_space()
    reachable_states = find_reachable_states(env)
    for state in state_list:
        neighbors, _ = env.get_neighbors(state)
        for next_state in neighbors:
            if next_state not in reachable_states:
                continue
            prob_str += f'From {state} to {next_state}, Prob = {env.transition_function(state, prm.TREATMENT_ACTION, next_state, [])}\n'
            prob_str += '\n\n'
    with open(out_file, prm.WRITE_FILE) as my_file:
        my_file.write(prob_str)


def add_state(env, curr_state, states):
    next_states = env.get_next_states(curr_state)
    for state in next_states:
        if not env.is_terminal_state(state) and state not in states:
            states.add(state)
            add_state(env, state, states)


def create_treatment_prob():
    treatment_prob = np.zeros((prm.NUM_FEATURES, 3))
    for feature in range(prm.NUM_FEATURES):
        treatment_prob[feature, :] = np.random.dirichlet(np.ones(3), 1)
    return treatment_prob


def calc_kap_meier(env: HazardEnv, policy_opt, policy_ref, group_size, seed, plot=True, group_name=None):
    kap_meier_opt = np.zeros(prm.HORIZON+1) + group_size
    kap_meier_ref = np.zeros(prm.HORIZON + 1) + group_size
    for i in range(group_size):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        time_of_death = 0
        while not is_terminal:
            next_state, reward, is_terminal = env.step(policy_opt[env.get_state_idx(curr_state)])
            curr_state = next_state
            time_of_death += 1
        for i in range(time_of_death, prm.HORIZON + 1):
            kap_meier_opt[i] -= 1
    for i in range(group_size):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        time_of_death = 0
        while not is_terminal:
            next_state, reward, is_terminal = env.step(policy_ref[env.get_state_idx(curr_state)])
            curr_state = next_state
            time_of_death += 1
        for i in range(time_of_death, prm.HORIZON + 1):
            kap_meier_ref[i] -= 1
    if plot:
        end_x_axis = 0
        y_axis_opt = list()
        y_axis_ref = list()
        while (kap_meier_opt[end_x_axis] != 0 or kap_meier_ref[end_x_axis] != 0) and end_x_axis != prm.HORIZON:
            y_axis_opt.append(kap_meier_opt[end_x_axis])
            y_axis_ref.append(kap_meier_ref[end_x_axis])
            end_x_axis += 1
        y_axis_opt.append(kap_meier_opt[end_x_axis])
        y_axis_ref.append(kap_meier_ref[end_x_axis])
        x_axis = list(range(0, end_x_axis + 1))
        folder_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + f'_seed_{seed}'
        curr_dir = f'./results/{folder_name}'
        os.mkdir(curr_dir)
        plt.style.use("ggplot")
        plt.figure()
        plt.step(x_axis, y_axis_opt, label='kaplan_meier opt')
        plt.step(x_axis, y_axis_ref, label='kaplan_meier ref')
        plt.ylabel('# Alive Patients')
        plt.xlabel('Time')
        plt.title(f'Kaplan Meier of {group_name} Group')
        plt.legend(loc='upper right')
        plt.savefig(f'{curr_dir}/{prm.KAP_MEIER_PLOT}')
        plt.show()
    return kap_meier_opt, kap_meier_ref

