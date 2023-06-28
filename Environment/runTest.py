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


def run_td_exp(env, policy):
    x_axis = np.linspace(prm.X_AXIS_LOWER_BOUND, prm.X_AXIS_UPPER_BOUND, prm.X_AXIS_RESOLUTION)
    td_prob = categorical_td(env, policy, x_axis)
    print(td_prob[state_to_idx_dict(env.get_start_state(), state_list)])
    plt.plot(x_axis, td_prob[state_to_idx_dict(env.get_start_state(), env.get_state_space())])
    plt.show()


def run_policy_evaluation(env, init_policy):
    optimal_policy = policy_iteration(env, init_policy)
    print_treatment_plan(env, state_list, optimal_policy)
    return optimal_policy


if __name__ == '__main__':
    control_group = create_control_data()
    env = HazardEnv(patient='Treatment', control_group=control_group)
    state_list = env.get_state_space()
    num_states = env.get_num_states()
    print(f'Num States = {num_states}')
    init_policy = np.zeros(env.get_num_states())
    for state in state_list:
        init_policy[state_to_idx_dict(state, env.get_state_space())] = prm.TREATMENT_ACTION
    optimal_policy = run_policy_evaluation(env, init_policy)
    run_td_exp(env, optimal_policy)
