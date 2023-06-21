import numpy as np
import Params as prm


def print_control_data (control_group):
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        print(f'Patient Number {i} Body Temp = {control_group[i][prm.BODY_TEMP["name"]]}')


def calculate_mean (control_data, feature):
    sum = 0
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        sum += np.array(control_data[i][feature["name"]]).sum()
    return sum/prm.SIZE_OF_CONTROL_GROUP


def state_to_idx_dict(feature):
    state_list = np.arange(feature['min_val'], feature['max_val'], feature['res'])
    idx_dict = dict()
    for idx, state in enumerate(state_list):
        idx_dict[state] = idx
    return idx_dict




