import matplotlib.pyplot as plt
import numpy as np
import Params as prm
from HazardEnv import HazardEnv
from CategoricalTD import categorical_td
from Utils import state_to_idx_dict
from Utils import create_control_data
from Utils import print_treatment_plan
from algorithm.PolicyOptimization import policy_evaluation, policy_iteration
from rich.traceback import install


install()


def run_td_exp():
    control_group = create_control_data()
    env = HazardEnv(patient='Treatment', control_group=control_group)
    state_list = env.get_state_space()
    print(state_list)
    policy = dict()
    for state in state_list:
        policy[state_to_idx_dict(state, env.get_state_space())] = prm.TREATMENT_ACTION
    x_axis = np.linspace(prm.X_AXIS_LOWER_BOUND, prm.X_AXIS_UPPER_BOUND, prm.X_AXIS_RESOLUTION)
    td_prob = categorical_td(env, policy, x_axis)
    print(td_prob[state_to_idx_dict(env.get_start_state(), state_list)])
    plt.plot(x_axis, td_prob[state_to_idx_dict(env.get_start_state(), env.get_state_space())])
    plt.show()


def run_policy_evaluation():
    control_group = create_control_data()
    env = HazardEnv(patient='Treatment', control_group=control_group)
    policy = dict()
    state_list = env.get_state_space()
    for state in state_list:
        policy[state_to_idx_dict(state, env.get_state_space())] = prm.TREATMENT_ACTION
    values = policy_evaluation(env, policy)
    start_state_idx = state_to_idx_dict(env.get_start_state(), state_list)
    optimal_policy = policy_iteration(env, policy)
    print_treatment_plan(state_list, optimal_policy)


if __name__ == '__main__':
    run_policy_evaluation()
