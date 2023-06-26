
import Params as prm
import numpy as np
from tqdm import tqdm
from HazardEnv import HazardEnv
from rich.traceback import install
install()


def print_control_data (control_group):
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        print(f'Patient Number {i} Body Temp = {control_group[i][prm.BODY_TEMP["name"]]}')


def state_to_idx_dict(state, state_space):
    return state_space.index(tuple(state))


def create_control_data():
    control_group = list()
    for patient in tqdm(range(prm.SIZE_OF_CONTROL_GROUP), desc='Simulating Control Group'):
        curr_env = HazardEnv(patient='Control')
        while not curr_env.is_terminal:
            curr_env.step(prm.CONTROL_ACTION)
        control_group.append(curr_env.get_history())
    return control_group


def print_treatment_plan(state_list, policy):
    for state_idx, action in policy.items():
        print(f'Under the Measurements:')
        state = state_list[state_idx]
        for idx, feature_value in enumerate(state):
            print(f'\t\t{prm.FEATURES_IDX_TO_NAME[idx]} = {feature_value}')
        print(f'{prm.ACTION_IDX_TO_NAME[action]}\n\n')
