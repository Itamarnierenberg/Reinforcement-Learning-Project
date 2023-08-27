import numpy as np
# Features Definition

# Body Temp
FEATURE_1 = {'idx': 0,
             'min_val': 35.0,
             'max_val': 43.0,
             'res': 0.5,
             'start_state': 40.0}

# Resting Pulse
FEATURE_2 = {'idx': 1,
             'min_val': 50,
             'max_val': 170,
             'res': 30,
             'start_state': 80.0}
# Fake
FEATURE_3 = {'idx': 3,
             'min_val': 1.0,
             'max_val': 2.0,
             'res': 0.1,
             'start_state': 1.5}

# Oxygen
FEATURE_4 = {'idx': 3,
             'min_val': 80.0,
             'max_val': 95.0,
             'res': 5,
             'start_state': 92.0}

# Respiratory
FEATURE_5 = {'idx': 4,
             'min_val': 8.0,
             'max_val': 30.0,
             'res': 5,
             'start_state': 15.0}


FEATURES = [FEATURE_1] # , FEATURE_2]#, FEATURE_3, FEATURE_4, FEATURE_5]
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
SIZE_OF_CONTROL_GROUP = 200 # 20000
SIZE_OF_ACTION_GROUP = 100 # 10000
HORIZON = 10
DISTANCE_FUNC = 'L1'
CONTROL_PROB = np.array([[0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1]])
# Randomize By Function TREATMENT_PROB = [0.1, 0.1, 0.8]
TIME_IDX = -1

# Policy Iteration Params
MAX_ITER = 100

# Categorical TD Params
NUM_EPOCHS_TD = 50000
STEP_SIZE = 0.1
DISCOUNT_FACTOR = 0.99
X_AXIS_LOWER_BOUND = -1000
X_AXIS_UPPER_BOUND = 1000
X_AXIS_RESOLUTION = 1000

# Monte Carlo Params
NUM_EPOCHS_MC = 5000000

# Q Learning Params
NUM_EPOCHS_Q = 500

# UCRL - MNL Parameters
NUM_EPOCHS_UCRL = 100
REG_PARAM = 0.2
CONF_RAD = 1

# Risk averse params
NUM_EPOCHS_RA = 50000

# File Settings
POLICY_OUTPUT_FILE = './Policy.txt'
PROB_OUPUT_FILE = './Prob.txt'
TD_MC_PLOT = 'td_mc_plot.png'
KAP_MEIER_PLOT = 'kap_meier_plot.png'
WRITE_FILE = 'w'
READ_FILE = 'r'

BAYES_MODE = True
NUM_INTERACTIONS = 3


# More states, a bigger state space with a more interesting cases
# Distributional Risk averse algorithms
# Always compare to MC