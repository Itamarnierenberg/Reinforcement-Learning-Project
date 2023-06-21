import numpy as np
import Params as prm


class HazardEnv:

    def __init__(self):
        self.num_states = 1
        for feature in prm.FEATURES:
            self.num_states *= len(np.arange(feature['min_val'], feature['max_val'], feature['res']))
        self.start_state = np.zeros(len(prm.FEATURES))
        self.curr_state = np.zeros(len(prm.FEATURES))
        for idx, feature in enumerate(prm.FEATURES):
            self.start_state[idx] = feature['start_state']
        self.curr_state = self.start_state

    def reset(self):
        self.curr_state = self.start_state

    def get_start_state(self):
        return self.start_state

    def get_state(self):
        return self.curr_state

    def is_terminal(self):
        for idx, feature in enumerate(prm.FEATURES):
            if self.curr_state[idx] >= feature['max_val'] or self.curr_state[idx] <= feature['min_val']:
                return True
        return False

    def transition_model(self, action):
        new_state_list = np.zeros(len(prm.FEATURES))
        if action == prm.CONTROL_ACTION:
            new_state_list[0] = self.curr_state[0] + np.random.choice([-0.5, 0, 0.5])     # What to do to Body Temperature Feature
        else:
            raise NotImplementedError
        return new_state_list

    def step(self, action):
        if self.is_terminal():
            return self.curr_state
        else:
            self.curr_state = self.transition_model(action)
            return self.curr_state

    def __str__(self):
        print_str = f'[INFO] Enviorment Information:\n'
        print_str += f'[INFO] Number of States = {self.num_states}\n'
        for idx, feature in enumerate(prm.FEATURES):
            print_str += f'[INFO] Feature = {feature["name"]}\n'
            print_str += f'[INFO] \t\tStart State = {self.start_state[idx]}\n'
            print_str += f'[INFO] \t\tCurrent State = {self.curr_state[idx]}\n'
            print_str += f'[INFO] \t\tIs Terminal = {self.is_terminal()}\n'
        return print_str


control_group = list(range(prm.SIZE_OF_CONTROL_GROUP))
for i in range(len(control_group)):
    control_group[i] = dict()
    control_group[i][prm.BODY_TEMP['name']] = list()
my_env = HazardEnv()
for patient in range(prm.SIZE_OF_CONTROL_GROUP):
    my_env.reset()
    while not my_env.is_terminal():
        for idx, feature in enumerate(prm.FEATURES):
            control_group[patient][feature['name']].append(my_env.get_state()[idx])
        my_env.step(prm.CONTROL_ACTION)
print(my_env)
for i in range(prm.SIZE_OF_CONTROL_GROUP):
    print(f'Patient Number {i} Body Temp = {control_group[i][prm.BODY_TEMP["name"]]}')