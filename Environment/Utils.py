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




