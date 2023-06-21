import matplotlib.pyplot as plt
import numpy as np
import Params as prm
from HazardEnv import HazardEnv
from CategoricalTD import categorical_td


def create_control_data():
    control_group = list(range(prm.SIZE_OF_CONTROL_GROUP))
    for i in range(prm.SIZE_OF_CONTROL_GROUP):
        control_group[i] = dict()
        for feature in prm.FEATURES:
            control_group[i][feature['name']] = list()

    my_env = HazardEnv(patient='Control')
    for patient in range(prm.SIZE_OF_CONTROL_GROUP):
        my_env.reset()
        i = 0
        while not my_env.is_terminal and i < prm.HORIZON:
            for idx, feature in enumerate(prm.FEATURES):
                control_group[patient][feature['name']].append(my_env.get_state()[idx])
            my_env.step(prm.CONTROL_ACTION)
            i = + 1
    return control_group


if __name__ == '__main__':
    control_group = create_control_data()
    env = HazardEnv(patient='Treatment', control_group=control_group)
    state_list = np.arange(prm.BODY_TEMP['min_val'], prm.BODY_TEMP['max_val'], prm.BODY_TEMP['res'])
    policy = dict()
    for state in state_list:
        policy[state] = prm.TREATMENT_ACTION
    x_axis = np.linspace(prm.X_AXIS_LOWER_BOUND, prm.X_AXIS_UPPER_BOUND, prm.X_AXIS_RESOLUTION)
    td_prob = categorical_td(env, policy, x_axis)
    plt.plot(x_axis, td_prob)
    plt.show()


