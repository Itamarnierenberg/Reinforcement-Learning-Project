from gymnasium import Env
from gymnasium.spaces import Discrete, Tuple
import numpy as np
import Params as prm
from ReplayMemory import ReplayMemory


# Init, Step, Render, Reset
class Patients(Env):

    def __init__(self, is_control=False, treatment_prob=None, control_group=None):
        """
        :param is_control: Is this patient part of the control group
        :param treatment_prob: If it is not in the control group what are the prob for simulation
        :param control_group: If it is not in the control group then expecting control group data for relative reward
        """
        # Creating the state space
        self.control_patient = is_control
        self.real_treatment_prob = treatment_prob
        self.history = list()
        self.control_group = control_group
        self.control_mean = self.calculate_control_mean()
        feature_list = list()
        for curr_feature in prm.FEATURES:
            low = curr_feature['min_val']
            high = curr_feature['max_val']
            num_elements = high - low + 1
            feature_disc = Discrete(num_elements, start=low)
            feature_list.append(feature_disc)
        feature_list = tuple(feature_list)
        self.observation_space = Tuple(feature_list)
        # Creating the action space
        self.action_space = Discrete(len(prm.ACTIONS))
        # Set start state
        self.state = []
        for idx, curr_feature in enumerate(prm.FEATURES):
            self.state.append(curr_feature['start_state'])
        self.state = tuple(self.state)
        # Set shower length
        self.trial_length = prm.HORIZON

    def render(self):
        # Implement visualisation
        pass

    def is_terminal_state(self) -> bool:
        """
        Checks if this patient has reached a terminal state
        :return:  True / False
        """
        for idx, feature in enumerate(prm.FEATURES):
            if self.state[idx] >= feature['max_val'] or self.state[idx] <= feature['min_val']:
                return True
        if self.trial_length <= 0:
            return True
        return False

    def step(self, action):
        """
        Tells a patient to take a specific action and view the results
        :param action: which action to take
        :return: next state, reward, is terminal state, info - unused
        """
        self.state = self.transition_model(action)
        self.trial_length -= 1
        if self.control_patient:
            reward = 0
        else:
            reward = self.calc_reward()
        done = self.is_terminal_state()
        self.history.append(self.state)

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def reset(self):
        """
        Reset the patient to his initial state
        :return: starting state
        """
        # Set start temp
        self.state = []
        for idx, curr_feature in enumerate(prm.FEATURES):
            self.state.append(curr_feature['start_state'])
        self.state = tuple(self.state)
        # Set shower length
        self.trial_length = prm.HORIZON
        return self.state

    def transition_model(self, action):
        new_state = np.zeros(len(prm.FEATURES))
        if action == prm.CONTROL_ACTION:
            for feature_idx in range(prm.NUM_FEATURES):
                step_list = [-1, 0, 1]
                new_state[feature_idx] = self.state[feature_idx] + np.random.choice(step_list, p=prm.CONTROL_PROB[feature_idx])     # What to do to Body Temperature Feature
        elif action == prm.TREATMENT_ACTION:
            for feature_idx in range(prm.NUM_FEATURES):
                step_list = [-1, 0, 1]
                new_state[feature_idx] = self.state[feature_idx] + np.random.choice(step_list, p=self.real_treatment_prob[feature_idx])
        else:
            raise NotImplementedError
        return new_state

    def calc_reward(self, state_input=None):
        time = prm.HORIZON - self.trial_length
        reward_arr = np.zeros(len(prm.FEATURES))
        for idx, feature in enumerate(prm.FEATURES):
            control_mean = self.control_mean[time][idx]
            if control_mean >= feature['max_val'] or control_mean <= feature['min_val']:
                reward_arr[idx] = 100
            elif self.state[idx] >= feature['max_val'] or self.state[idx] <= feature['min_val']:
                reward_arr[idx] = -100
            else:
                hazard_ratio = Patients.distance_func(self.state[idx], feature['max_val'], feature['min_val']) / \
                                Patients.distance_func(control_mean, feature['max_val'], feature['min_val'])
                if hazard_ratio > 1:
                    reward_arr[idx] = hazard_ratio
                elif hazard_ratio == 1:
                    reward_arr[idx] = 0
                else:
                    reward_arr[idx] = -(1/hazard_ratio)
        return np.sum(reward_arr) / prm.NUM_FEATURES

    @staticmethod
    def distance_func(x, x_max, x_min):
        if prm.DISTANCE_FUNC == 'L1':
            min_list = np.array([np.abs(x - x_max), np.abs(x - x_min)])
            return np.min(min_list)

        elif prm.DISTANCE_FUNC == 'L2':
            return np.min([np.abs(x - x_max), np.abs(x - x_min)])
        else:
            raise NotImplementedError

    def calculate_control_mean(self):
        if self.control_patient:
            return None
        sum_arr = np.zeros((prm.HORIZON + 1, len(prm.FEATURES)))
        count_arr = np.zeros(prm.HORIZON + 1)
        for patient in self.control_group:
            for curr_time, state in enumerate(patient):
                sum_arr[curr_time] += state
                count_arr[curr_time] += 1
        for curr_time in range(prm.HORIZON + 1):
            if count_arr[curr_time] == 0:
                continue
            sum_arr[curr_time] = sum_arr[curr_time] / count_arr[curr_time]
        return sum_arr

    def get_history(self):
        return self.history
