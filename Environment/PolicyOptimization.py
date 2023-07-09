import Environment.HazardEnv as HazardEnv
import Environment.Params as prm
import numpy as np
import math


def policy_evaluation(env, policy, discount_factor=prm.DISCOUNT_FACTOR, epsilon=0.01, treatment_prob=prm.TREATMENT_PROB) :
    values = np.zeros(env.get_num_states())
    state_list = env.get_state_space()
    while True:
        delta = 0
        for state in state_list:
            state_idx = env.get_state_idx(state)
            old_values = values
            next_states, rewards = env.get_neighbors(state)
            curr_value = 0
            for i in range(len(next_states)):
                next_state_idx = env.get_state_idx(next_states[i])
                prob = env.transition_function(state, policy[state_idx], next_states[i], treatment_prob)
                curr_value += prob * (rewards[i] + discount_factor * values[next_state_idx])
            values[state_idx] = curr_value
            delta = np.max([delta, np.abs(old_values[state_idx] - values[state_idx])])
        if delta < epsilon:
            return values


# Treatment prob is added because we dont know the real prob so we use the one we have at that moment
def policy_iteration(env, initial_policy, discount_factor=prm.DISCOUNT_FACTOR, max_iter=prm.MAX_ITER, treatment_prob=prm.TREATMENT_PROB):
    state_list = env.get_state_space()
    policy = initial_policy
    num_iter = 0
    while True:
        values = policy_evaluation(env, policy, treatment_prob=treatment_prob)
        is_unchanged = True
        print(f'Iteration Number = {num_iter}')
        for state in state_list:
            max_action_util = -math.inf
            max_action = None
            state_idx = env.get_state_idx(state)
            next_states, rewards = env.get_neighbors(state)
            for action in prm.ACTIONS:
                curr_util = 0
                for i in range(len(next_states)):
                    next_state_idx = env.get_state_idx(next_states[i])
                    prob = env.transition_function(state, action, next_states[i], treatment_prob)
                    curr_util += prob * (rewards[i] + discount_factor * values[next_state_idx])
                if curr_util > max_action_util:
                    max_action_util = curr_util
                    max_action = action
            curr_util = 0
            for i in range(len(next_states)):
                next_state_idx = env.get_state_idx(next_states[i])
                prob = env.transition_function(state, policy[state_idx], next_states[i], treatment_prob)
                curr_util += prob * (rewards[i] + discount_factor * values[next_state_idx])
            if max_action_util > curr_util:
                policy[state_idx] = max_action
                is_unchanged = False
        num_iter += 1
        if is_unchanged or num_iter == max_iter:
            return policy


def perform_interactions(env, policy):
    is_terminal = env.is_terminal
    curr_state = env.get_state()
    for interaction in range(prm.NUM_INTERACTIONS):
        if is_terminal:
            break
        state_idx = env.get_state_idx(curr_state)
        next_state, reward, is_terminal = env.step(policy[state_idx])
        next_state_idx = env.get_state_idx(next_state)
        new_params = env.get_dist_params()
        new_params[state_idx, next_state_idx] += 1
        env.update_dist_params(new_params)
        curr_state = next_state


def ucrl_mnl(env, num_epochs=prm.NUM_EPOCHS_UCRL, regular_prm=prm.REG_PARAM, conf_rad=prm.CONF_RAD):
    feature_map, dim = env.get_feature_map
    ident = np.eye(dim)
    A_1 = regular_prm * ident
    theta = np.zeros(dim)
    state_list = env.get_state_space()
    Q = np.array(len(state_list), prm.NUM_ACTIONS)
    for epoch in num_epochs:
        # for each h in H set Q_k,h as described, using theta_k and beta_k
        for state in state_list:
            state_idx = env.get_state_idx(state)
            for action in prm.ACTIONS:
                curr_reward = env.calc_reward(state)
                next_states, _ = env.get_neighbors(state)
                curr_util = 0
                denum = 0
                for next_state in next_states:
                    next_state_idx = env.get_state_idx(next_state)
                    denum += np.exp(feature_map[state_idx, action, next_state_idx].T*theta)
                for next_state in next_states:
                    value = np.max(Q[next_state, :])
                    curr_util += np.exp(feature_map[state_idx, action, next_state_idx].T*theta) * value / denum
                # regular_term = 2 * prm.HORIZON * conf_rad FIXME
                Q[state_idx, action] = curr_reward + curr_util
        is_terminal = False





