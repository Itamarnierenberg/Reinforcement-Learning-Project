import numpy as np
import Params as prm
import itertools


class HazardEnv:

    def __init__(self, patient, control_group=None, real_treatment_prob=None):
        feature_space_list = list()
        for idx, feature in enumerate(prm.FEATURES):
            feature_space_list.append(
                np.arange(feature['min_val'], feature['max_val'] + feature['res'], feature['res']))
        feature_space_list.append(np.arange(0, prm.HORIZON + 1, 1))
        self.state_space = list(itertools.product(*feature_space_list))
        self.num_states = len(self.state_space)
        if patient == 'Treatment':
            assert control_group is not None
            self.control_group = control_group
            self.control_mean = self.calculate_control_mean()
            self.dist_params = np.zeros((self.num_states, self.num_states)) + 1/self.num_states
            self.real_treatment_prob = real_treatment_prob
        elif patient != 'Control':
            print('Patient can be only Control or Treatment')
            raise NotImplementedError
        self.patient = patient
        self.start_state = np.zeros(len(prm.FEATURES) + 1)
        self.curr_state = np.zeros(len(prm.FEATURES) + 1)
        self.curr_time = 0
        for idx, feature in enumerate(prm.FEATURES):
            self.start_state[idx] = feature['start_state']
        self.curr_state = self.start_state
        self.is_terminal = False
        self.history = list()
        self.history.append(self.curr_state)

    def reset(self):
        self.curr_state = self.start_state
        self.curr_time = 0
        self.is_terminal = self.is_terminal_state()

    def get_start_state(self):
        return self.start_state

    def get_state(self):
        return self.curr_state

    def get_num_states(self):
        return self.num_states

    def get_time(self):
        raise NotImplementedError
        # return self.curr_time

    def get_state_space(self):
        return self.state_space

    def get_history(self):
        return self.history

    def get_dist_params(self):
        return self.dist_params

    def is_terminal_state(self, state=None):
        if state is None:
            state = self.curr_state
        for idx, feature in enumerate(prm.FEATURES):
            if state[idx] >= feature['max_val'] or state[idx] <= feature['min_val']:
                return True
        if state[prm.TIME_IDX] >= prm.HORIZON:
            return True
        return False

    def transition_model(self, action):
        new_state = np.zeros(len(prm.FEATURES) + 1)
        if action == prm.CONTROL_ACTION:
            for feature_idx in range(prm.NUM_FEATURES):
                step_list = [-prm.FEATURES[feature_idx]['res'], 0, prm.FEATURES[feature_idx]['res']]
                new_state[feature_idx] = self.curr_state[feature_idx] + np.random.choice(step_list, p=prm.CONTROL_PROB[feature_idx])     # What to do to Body Temperature Feature
        elif action == prm.TREATMENT_ACTION:
            for feature_idx in range(prm.NUM_FEATURES):
                step_list = [-prm.FEATURES[feature_idx]['res'], 0, prm.FEATURES[feature_idx]['res']]
                new_state[feature_idx] = self.curr_state[feature_idx] + np.random.choice(step_list, p=self.real_treatment_prob[feature_idx])
        else:
            raise NotImplementedError
        return new_state

    def step(self, action):
        if self.is_terminal:
            raise NotImplementedError
        else:
            self.curr_state = self.transition_model(action)
            self.curr_time += 1
            self.curr_state[prm.TIME_IDX] = self.curr_time
            if self.patient == 'Control':
                reward = 0
            else:
                reward = self.calc_reward()
            self.is_terminal = self.is_terminal_state()
            self.history.append(self.curr_state)
            return self.curr_state, reward, self.is_terminal

    def get_next_states(self, state=None):
        if state is None:
            state = self.curr_state
        feature_space_list = list()
        for idx, feature in enumerate(prm.FEATURES):
            new_state_left = np.max([state[idx] - feature['res'], feature['min_val']])
            new_state_right = np.min([state[idx] + feature['res'], feature['max_val']])
            feature_space_list.append([new_state_left, state[idx], new_state_right])
        feature_space_list.append([state[prm.TIME_IDX] + 1])
        next_states = list(itertools.product(*feature_space_list))
        return next_states

    def get_neighbors(self, state=None, calc_rewards=True):
        if self.is_terminal_state(state):
            return [], []
        next_states = self.get_next_states(state)
        if calc_rewards:
            rewards = [self.calc_reward(next_state) for next_state in next_states]
            return next_states, rewards
        else:
            return next_states

    def transition_function(self, state, action, next_state, treatment_prob):
        next_states, _ = self.get_neighbors(state)
        if next_state not in next_states:
            return 0
        if action == prm.CONTROL_ACTION:
            prob_list = prm.CONTROL_PROB
        elif action == prm.TREATMENT_ACTION:
            if prm.BAYES_MODE:
                state_idx = self.get_state_idx(state)
                next_state_idx = self.get_state_idx(next_state)
                denum = np.sum(self.dist_params[state_idx, :])
                if denum == 0:
                    return 0
                return self.dist_params[state_idx, next_state_idx] / np.sum(self.dist_params[state_idx, :])
            else:
                prob_list = treatment_prob
        else:
            raise NotImplementedError
        transition_prob = None
        for feature in prm.FEATURES:
            change_diff = state[feature['idx']] - next_state[feature['idx']]
            if change_diff == feature['res']:
                if transition_prob is None:
                    transition_prob = prob_list[feature['idx']][0]
                else:
                    transition_prob *= prob_list[feature['idx']][0]
            elif change_diff == 0:
                if transition_prob is None:
                    transition_prob = prob_list[feature['idx']][1]
                else:
                    transition_prob *= prob_list[feature['idx']][1]
            elif change_diff == -feature['res']:
                if transition_prob is None:
                    transition_prob = prob_list[feature['idx']][2]
                else:
                    transition_prob *= prob_list[feature['idx']][2]
            else:
                raise ValueError(f'The feature {feature} increased or decreased by more then its resoultion in a single step.'
                                 f'\n\n Current State = {state}, Next State = {next_state}, Change = {change_diff}')
        return transition_prob

    @staticmethod
    def distance_func(x, x_max, x_min):
        if prm.DISTANCE_FUNC == 'L1':
            min_list = np.array([np.abs(x - x_max), np.abs(x - x_min)])
            return np.min(min_list)

        elif prm.DISTANCE_FUNC == 'L2':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])
        else:
            raise NotImplementedError

    def calc_reward(self, state_input=None):
        time = self.curr_time
        if state_input is not None:
            state = state_input
            time = state_input[prm.TIME_IDX]
        else:
            state = self.curr_state
        reward_arr = np.zeros(len(prm.FEATURES))
        for idx, feature in enumerate(prm.FEATURES):
            control_mean = self.control_mean[time][idx]
            if control_mean >= feature['max_val'] or control_mean <= feature['min_val']:
                reward_arr[idx] = 100
            elif state[idx] >= feature['max_val'] or state[idx] <= feature['min_val']:
                reward_arr[idx] = -100
            else:
                hazard_ratio = HazardEnv.distance_func(state[idx], feature['max_val'], feature['min_val']) / \
                                HazardEnv.distance_func(control_mean, feature['max_val'], feature['min_val'])
                if hazard_ratio > 1:
                    reward_arr[idx] = hazard_ratio
                elif hazard_ratio == 1:
                    reward_arr[idx] = 0
                else:
                    reward_arr[idx] = -(1/hazard_ratio)
        return np.sum(reward_arr) / prm.NUM_FEATURES

    def calculate_control_mean(self):
        sum_arr = np.zeros((prm.HORIZON + 1, len(prm.FEATURES)))
        count_arr = np.zeros(prm.HORIZON + 1)
        for patient in range(prm.SIZE_OF_CONTROL_GROUP):
            for state in self.control_group[patient]:
                curr_time = int(state[prm.TIME_IDX])
                sum_arr[curr_time] += state[:prm.TIME_IDX]
                count_arr[curr_time] += 1
        for curr_time in range(prm.HORIZON + 1):
            if count_arr[curr_time] == 0:
                continue
            sum_arr[curr_time] = sum_arr[curr_time] / count_arr[curr_time]
        return sum_arr

    def get_feature_map(self):
        dim = (prm.NUM_FEATURES + 1) * prm.NUM_ACTIONS * (prm.NUM_FEATURES + 1)
        feature_map = np.array((len(self.state_space), prm.NUM_ACTIONS, len(self.state_space), dim))
        for state in self.state_space:
            state_idx = self.get_state_idx(state)
            for action in prm.ACTIONS:
                for next_state in self.state_space:
                    feature_vec = np.array(dim)
                    next_state_idx = self.get_state_idx(next_state)
                    moving_idx = 0
                    for feature_idx, feature in enumerate(prm.FEATURES):
                        feature_vec[moving_idx] = state[feature_idx]
                        moving_idx += 1
                    feature_vec[moving_idx] = state[prm.TIME_IDX]
                    moving_idx += 1
                    feature_vec[moving_idx] = action
                    moving_idx += 1
                    for feature_idx, feature in enumerate(prm.FEATURES):
                        feature_vec[moving_idx] = next_state[feature_idx]
                        moving_idx += 1
                    feature_vec[moving_idx] = next_state[prm.TIME_IDX]
                    feature_map[state_idx, action, next_state_idx] = feature_vec
        return feature_map, dim

    def get_state_idx(self, state):
        return self.state_space.index(tuple(state))

    def update_dist_params(self, new_dist_params):
        self.dist_params = new_dist_params

    def __str__(self):
        print_str = f'[INFO] Environment Information:\n'
        print_str += f'[INFO] Number of States = {self.num_states}\n'
        print_str += f'[INFO] Current Time = {self.curr_time}\n'
        print_str += f'[INFO] Is Terminal = {self.is_terminal}\n'
        for idx, feature in enumerate(prm.FEATURES):
            print_str += f'[INFO] Feature = {feature["name"]}\n'
            print_str += f'[INFO] \t\tStart State = {self.start_state[idx]}\n'
            print_str += f'[INFO] \t\tCurrent State = {self.curr_state[idx]}\n'
        return print_str


# stable-base-lines
# LLMs
# SARSA for exploration explotation
