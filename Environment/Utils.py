
from HazardEnv import HazardEnv
import Params as prm
def create_control_data():
    control_group = list(range(prm.SIZE_OF_CONTROL_GROUP))
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        control_group[i] = dict()
        for feature in prm.FEATURES:
            control_group[i][feature['name']] = list()

    my_env = HazardEnv()
    for patient in range(prm.SIZE_OF_CONTROL_GROUP):
        my_env.reset()
        i = 0
        while not my_env.is_terminal() and i < prm.HORIZON:
            for idx, feature in enumerate(prm.FEATURES):
                control_group[patient][feature['name']].append(my_env.get_state()[idx])
            my_env.step(prm.CONTROL_ACTION)
            i = + 1
    return control_group

def print_control_data (control_group):
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        print(f'Patient Number {i} Body Temp = {control_group[i][prm.BODY_TEMP["name"]]}')

def calculate_mean (control_data, feature, time):
    sum = 0
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        sum += control_data[i][feature["name"]][time]
    return sum/prm.SIZE_OF_CONTROL_GROUP


def state_to_idx_dict(feature):
    state_list = np.arange(feature['min_val'], feature['max_val'], feature['res'])
    idx_dict = dict()
    for idx, state in enumerate(state_list):
        idx_dict[state] = idx
    return idx_dict




