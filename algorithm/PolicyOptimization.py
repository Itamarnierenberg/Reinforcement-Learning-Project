import Environment.HazardEnv as HazardEnv
import Environment.Params as prm
from Environment.Utils import state_to_idx_dict
import numpy as np
import math


def policy_evaluation(env, policy, discount_factor=prm.DISCOUNT_FACTOR, epsilon=0.01) :
    values = np.zeros(env.get_num_states())
    state_list = env.get_state_space()
    while True:
        delta = 0
        for state in state_list:
            state_idx = state_to_idx_dict(state, state_list)
            old_values = values
            next_states, rewards = env.get_neighbors(state)
            curr_value = 0
            for i in range(len(next_states)):
                next_state_idx = state_to_idx_dict(next_states[i], state_list)
                prob = env.transition_function(state, policy[state_idx], next_states[i])
                curr_value += prob * (rewards[i] + discount_factor * values[next_state_idx])
            values[state_idx] = curr_value
            delta = np.max([delta, np.abs(old_values[state_idx] - values[state_idx])])
        if delta < epsilon:
            return values


def policy_iteration(env, initial_policy, discount_factor=prm.DISCOUNT_FACTOR):
    state_list = env.get_state_space()
    policy = initial_policy
    num_iter = 0
    while True:
        values = policy_evaluation(env, policy)
        is_unchanged = True
        print(f'Iteration Number = {num_iter}')
        for state in state_list:
            max_action_util = -math.inf
            max_action = None
            state_idx = state_to_idx_dict(state, state_list)
            next_states, rewards = env.get_neighbors(state)
            for action in prm.ACTIONS:
                curr_util = 0
                for i in range(len(next_states)):
                    next_state_idx = state_to_idx_dict(next_states[i], state_list)
                    prob = env.transition_function(state, action, next_states[i])
                    curr_util += prob * (rewards[i] + discount_factor * values[next_state_idx])
                    if curr_util > max_action_util:
                        max_action_util = curr_util
                        max_action = action
            curr_util = 0
            for i in range(len(next_states)):
                next_state_idx = state_to_idx_dict(next_states[i], state_list)
                prob = env.transition_function(state, policy[state_idx], next_states[i])
                curr_util += prob * (rewards[i] + discount_factor * values[next_state_idx])
                print(max_action_util)
                print(curr_util)
                if max_action_util > curr_util:
                    policy[state_idx] = max_action
                    is_unchanged = False
        num_iter += 1
        if num_iter == 1000:
            print('hi')
        if is_unchanged:
            return policy


