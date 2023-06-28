SUC_PROB = 0.5
PROBS = [SUC_PROB, 1-SUC_PROB]
HORIZON = 50
import Params as prm

import random


def get_next_state(current_state):
    next_states = [state + BODY_TEMP[res], state - BODY_TEMP[res]]
    return random.choices(next_states,PROBS)[0]

def genrate_data (samples_num = 100):
    control_data= []
    i=0
    while (i != samples_num) :
        state = START_STATE
        trajectory= []
        trajectory.append(state)
        while (BODY_TEMP[min_val]<= START_STATE <= BODY_TEMP[max_val] and len(trajectory) < = HORIZON):
            next_stat = get_next_state(state)
            trajectory.append(next_stat)
            state=next_stat
        control_data.append(trajectory)
        i+=1
    return np.array(control_data)
def calculate_mean (control_data,time):
    return control_data[time].mean()

