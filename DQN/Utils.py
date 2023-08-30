from tqdm import tqdm
import numpy as np
from gymnasium.spaces import Discrete, Tuple
import Params as prm
from datetime import datetime
import matplotlib.pyplot as plt
import os
from Patients import Patients


def print_control_data(control_group):
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        for feature in prm.FEATURES:
            print(f'Patient Number {i} Body Temp = {control_group[i][feature["name"]]}')


def create_control_data():
    control_group = list()
    d_list = np.zeros(prm.HORIZON) + prm.SIZE_OF_CONTROL_GROUP
    for patient in tqdm(range(prm.SIZE_OF_CONTROL_GROUP), desc='Simulating Control Group'):
        curr_env = Patients(is_control=True)
        done = False
        time_of_d = 0
        while not done:
            _, _, done, _ = curr_env.step(prm.CONTROL_ACTION)
            time_of_d += 1
        d_list[time_of_d:] -= 1
        control_group.append(curr_env.get_history())
    return control_group, d_list

def create_treatment_prob():
    treatment_prob = np.zeros((prm.NUM_FEATURES, 3))
    for feature in range(prm.NUM_FEATURES):
        treatment_prob[feature, :] = np.random.dirichlet(np.ones(3), 1)
    return treatment_prob
