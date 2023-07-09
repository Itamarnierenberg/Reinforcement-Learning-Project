
import Params as prm
import numpy as np
from tqdm import tqdm
from HazardEnv import HazardEnv
from rich.traceback import install
from datetime import datetime
install()


def print_control_data (control_group):
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        print(f'Patient Number {i} Body Temp = {control_group[i][prm.BODY_TEMP["name"]]}')


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


def print_learned_dist(env, out_file=prm.PROB_OUPUT_FILE):
    prob_str = ''
    state_list = env.get_state_space()
    for state in state_list:
        neighbors, _ = env.get_neighbors(state)
        for next_state in neighbors:
            prob_str += f'From {state} to {next_state}, Prob = {env.transition_function(state, prm.TREATMENT_ACTION, next_state, [])}\n'
        prob_str += '\n\n'
    with open(out_file, prm.WRITE_FILE) as my_file:
        my_file.write(prob_str)