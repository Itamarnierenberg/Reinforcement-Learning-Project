import numpy as np
# Features Definition

# Body Temp
FEATURE_1 = {
             'min_val': 0,
             'max_val': 20,
             'start_state': 10}

# Resting Pulse
FEATURE_2 = {
             'min_val': 50,
             'max_val': 170,
             'start_state': 80}
# Fake
FEATURE_3 = {
             'min_val': 1,
             'max_val': 8,
             'start_state': 2}

# Oxygen
FEATURE_4 = {
             'min_val': 80,
             'max_val': 95,
             'start_state': 92.0}

# Respiratory
FEATURE_5 = {
             'min_val': 8,
             'max_val': 30,
             'start_state': 15}


FEATURES = [FEATURE_1, FEATURE_2]#, FEATURE_3, FEATURE_4, FEATURE_5]
NUM_FEATURES = len(FEATURES)

FEATURES_IDX_TO_NAME = [f'Feature {i}' for i in range(NUM_FEATURES)]
FEATURES_IDX_TO_NAME.append('Time')

# Actions Definition
CONTROL_ACTION = 0
TREATMENT_ACTION = 1
ACTIONS = [CONTROL_ACTION, TREATMENT_ACTION]
NUM_ACTIONS = len(ACTIONS)
ACTION_IDX_TO_NAME = ["Don't take a pill", "Take a pill"]

# Environment Params
SIZE_OF_CONTROL_GROUP = 50
SIZE_OF_ACTION_GROUP = 50
HORIZON = 50
DISTANCE_FUNC = 'L1'
CONTROL_PROB = np.array([[0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1]])


# File Settings
POLICY_OUTPUT_FILE = './Policy.txt'
PROB_OUPUT_FILE = './Prob.txt'
TD_MC_PLOT = 'td_mc_plot.png'
KAP_MEIER_PLOT = 'kap_meier_plot.png'
WRITE_FILE = 'w'
READ_FILE = 'r'
