import numpy as np
import Params as prm
import itertools


class HazardEnv:

    def __init__(self, patient, control_group=None):
        if patient == 'Treatment':
            assert control_group is not None
            self.control_group = control_group
            self.control_mean = self.calculate_control_mean()
        elif patient != 'Control':
            print('Patient can be only Control or Treatment')
            raise NotImplementedError
        feature_space_list = list()
        for idx, feature in enumerate(prm.FEATURES):
            feature_space_list.append(np.arange(feature['min_val'], feature['max_val'] + feature['res'], feature['res']))
        feature_space_list.append(np.arange(0, prm.HORIZON + 1, 1))
        self.state_space = list(itertools.product(*feature_space_list))
        self.patient = patient
        self.num_states = len(self.state_space)
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
            step_list = [-prm.FEATURES[0]['res'], 0, prm.FEATURES[0]['res']]
            new_state[0] = self.curr_state[0] + np.random.choice(step_list, p=prm.CONTROL_PROB)     # What to do to Body Temperature Feature
        elif action == prm.TREATMENT_ACTION:
            step_list = [-prm.FEATURES[0]['res'], 0, prm.FEATURES[0]['res']]
            new_state[0] = self.curr_state[0] + np.random.choice(step_list, p=prm.TREATMENT_PROB)
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
            feature_space_list.append([new_state_left, new_state_right])
        feature_space_list.append([state[prm.TIME_IDX] + 1])
        next_states = list(itertools.product(*feature_space_list))
        return next_states

    def get_neighbors(self, state=None):
        if self.is_terminal_state(state):
            return [], []
        next_states = self.get_next_states(state)
        rewards = [self.calc_reward(next_state) for next_state in next_states]
        return next_states, rewards

    def transition_function(self, state, action, next_state, treatment_prob=prm.TREATMENT_PROB):
        next_states, _ = self.get_neighbors(state)
        if next_state not in next_states:
            return 0
        if action == prm.CONTROL_ACTION:
            prob_list = prm.CONTROL_PROB
        elif action == prm.TREATMENT_ACTION:
            prob_list = treatment_prob
        else:
            raise NotImplementedError
        if state[prm.BODY_TEMP['idx']] < next_state[prm.BODY_TEMP['idx']]:
            return prob_list[2]
        elif state[prm.BODY_TEMP['idx']] == next_state[prm.BODY_TEMP['idx']]:
            return prob_list[1]
        else:
            return prob_list[0]

    @staticmethod
    def distance_func(x, x_max, x_min):
        if prm.DISTANCE_FUNC == 'L1':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])

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
                reward_arr[idx] = 1
            elif state[idx] >= feature['max_val'] or state[idx] <= feature['min_val']:
                reward_arr[idx] = -5
            else:
                hazard_ratio = HazardEnv.distance_func(state[idx], feature['max_val'], feature['min_val']) / \
                                HazardEnv.distance_func(control_mean, feature['max_val'], feature['min_val'])
                if hazard_ratio > 1:
                    reward_arr[idx] = 1/hazard_ratio
                elif hazard_ratio == 1:
                    reward_arr[idx] = 0
                else:
                    reward_arr[idx] = -hazard_ratio
        return reward_arr

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
