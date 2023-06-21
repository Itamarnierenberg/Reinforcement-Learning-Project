import numpy as np
import Params as prm
from Utils import calculate_mean


class HazardEnv:

    def __init__(self, patient, control_group=None):
        if patient == 'Treatment':
            assert control_group is not None
            self.control_group = control_group
        elif patient != 'Control':
            print('Patient can be only Control or Treatment')
            raise NotImplementedError
        self.patient = patient
        self.num_states = 1
        for feature in prm.FEATURES:
            self.num_states *= len(np.arange(feature['min_val'], feature['max_val'], feature['res']))
        self.num_states *= prm.HORIZON
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
            self.curr_state[-1] += 1
            if self.patient == 'Control':
                reward = 0
            else:
                reward = self.calc_reward()
            return self.curr_state, reward, self.is_terminal
        else:
            self.curr_state = self.transition_model(action)
            self.curr_state[-1] += 1
            if self.patient == 'Control':
                reward = 0
            else:
                reward = self.calc_reward()
            self.update_terminal()
            return self.curr_state, reward, self.is_terminal

    @staticmethod
    def distance_func(x, x_max, x_min):
        if prm.DISTANCE_FUNC == 'L1':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])

        elif prm.DISTANCE_FUNC == 'L2':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])
        else:
            raise NotImplementedError

    def calc_reward(self):
        reward_arr = np.zeros(len(prm.FEATURES))
        for idx, feature in enumerate(prm.FEATURES):
            control_mean = calculate_mean(self.control_group, feature)
            hazard_ratio = HazardEnv.distance_func(self.curr_state[idx], feature['max_val'], feature['min_val']) / \
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
