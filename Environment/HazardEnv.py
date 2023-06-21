import numpy as np
import Params as prm
from Utils import calculate_mean
import itertools


class HazardEnv:

    def __init__(self, patient, control_group=None):
        if patient == 'Treatment':
            assert control_group is not None
            self.control_group = control_group
        elif patient != 'Control':
            print('Patient can be only Control or Treatment')
            raise NotImplementedError
        feature_space_list = list()
        for idx, feature in enumerate(prm.FEATURES):
            feature_space_list.append(np.arange(feature['min_val'], feature['max_val'], feature['res']))
        feature_space_list.append(np.arange(0, prm.HORIZON, 1))
        self.state_space = list(itertools.product(*feature_space_list))
        self.patient = patient
        self.num_states = 1
        for feature in prm.FEATURES:
            self.num_states *= len(np.arange(feature['min_val'], feature['max_val'], feature['res']))
        self.start_state = np.zeros(len(prm.FEATURES) + 1)
        self.curr_state = np.zeros(len(prm.FEATURES) + 1)
        for idx, feature in enumerate(prm.FEATURES):
            self.start_state[idx] = feature['start_state']
        self.curr_state = self.start_state
        self.is_terminal = False

    def reset(self):
        self.curr_state = self.start_state
        self.is_terminal = self.update_terminal()

    def get_start_state(self):
        return self.start_state

    def get_state(self):
        return self.curr_state

    def get_num_states(self):
        return self.num_states

    def update_terminal(self):
        for idx, feature in enumerate(prm.FEATURES):
            if self.curr_state[idx] >= feature['max_val'] or self.curr_state[idx] <= feature['min_val']:
                self.is_terminal = True
                return True
        self.is_terminal = False
        return False

    def transition_model(self, action):
        new_state_list = np.zeros(len(prm.FEATURES))
        if action == prm.CONTROL_ACTION:
            new_state_list[0] = self.curr_state[0] + np.random.choice(prm.RES, p=prm.CONTROL_PROB)     # What to do to Body Temperature Feature
        elif action == prm.TREATMENT_ACTION:
            new_state_list[0] = self.curr_state[0] + np.random.choice(prm.RES, p=prm.TREATMENT_PROB)
        else:
            raise NotImplementedError
        return new_state_list

    def step(self, action):
        if self.is_terminal:
            # self.curr_state[-1] += 1
            # if self.patient == 'Control':
            #     reward = 0
            # else:
            #     reward = self.calc_reward()
            # return self.curr_state, reward, self.is_terminal
            raise NotImplementedError
        else:
            self.curr_state = self.transition_model(action)
            self.curr_state[-1] += 1
            if self.patient == 'Control':
                reward = 0
            else:
                reward = self.calc_reward()
            self.update_terminal()
            return self.curr_state, reward, self.is_terminal

    def get_next_states (self, feature):
        next_states = []
        feature_idx = prm.FEATURES.index(feature)
        curr_state = self.curr_state[feature_idx]
        time =  self.curr_state[-1]
        next_states.append(np.max(curr_state-0.5), prm.feature["min_val"])
        next_states.append(curr_state)
        next_states.append(np.min(curr_state+0.5), prm.feature["max_val"])
        return next_states
    def get_neighbors (self, feature):
        next_states = self.get_next_states(feature)
        time = self.curr_state[-1] + 1
        rewards = [self.calc_reward(state, time) for state in next_states]
        return next_states, rewards



    @staticmethod
    def distance_func(x, x_max, x_min):
        if prm.DISTANCE_FUNC == 'L1':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])

        elif prm.DISTANCE_FUNC == 'L2':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])
        else:
            raise NotImplementedError

    def calc_reward(self, state = curr.state):
        reward_arr = np.zeros(len(prm.FEATURES))
        for idx, feature in enumerate(prm.FEATURES):
            control_mean = calculate_mean(self.control_group, feature, self.state[-1])
            hazard_ratio = HazardEnv.distance_func(self.state[idx], feature['max_val'], feature['min_val']) / \
                           HazardEnv.distance_func(control_mean, feature['max_val'], feature['min_val'])
            if hazard_ratio > 1:
                reward_arr[idx] = 1
            if hazard_ratio == 1:
                reward_arr[idx] = 0
            else:
                reward_arr[idx] = -1
        return reward_arr

    def __str__(self):
        print_str = f'[INFO] Enviorment Information:\n'
        print_str += f'[INFO] Number of States = {self.num_states}\n'
        for idx, feature in enumerate(prm.FEATURES):
            print_str += f'[INFO] Feature = {feature["name"]}\n'
            print_str += f'[INFO] \t\tStart State = {self.start_state[idx]}\n'
            print_str += f'[INFO] \t\tCurrent State = {self.curr_state[idx]}\n'
            print_str += f'[INFO] \t\tIs Terminal = {self.is_terminal()}\n'
        return print_str
