# Features Definition

BODY_TEMP = {'idx': 0,
             'min_val': 35.0,
             'max_val': 43.0,
             'res': 0.5,
             'start_state': 40.0}

FEATURES = [BODY_TEMP]
NUM_FEATURES = len(FEATURES)

FEATURES_IDX_TO_NAME = ['Body Temperature', 'Time']

# Actions Definition
CONTROL_ACTION = 0
TREATMENT_ACTION = 1
ACTIONS = [CONTROL_ACTION, TREATMENT_ACTION]
NUM_ACTIONS = len(ACTIONS)
ACTION_IDX_TO_NAME = ["Don't take a pill", "Take a pill"]

# Environment Params
SIZE_OF_CONTROL_GROUP = 100
SIZE_OF_ACTION_GROUP = 100
HORIZON = 20
DISTANCE_FUNC = 'L1'
CONTROL_PROB = [0.8, 0.1, 0.1]
TREATMENT_PROB = [0.1, 0.1, 0.8]
TIME_IDX = -1

# Policy Iteration Params
MAX_ITER = 100

# Categorical TD Params
NUM_EPOCHS_TD = 5000
STEP_SIZE = 0.1
DISCOUNT_FACTOR = 0.99
X_AXIS_LOWER_BOUND = -100
X_AXIS_UPPER_BOUND = 100
X_AXIS_RESOLUTION = 100

# Monte Carlo Params
NUM_EPOCHS_MC = 500000

# Q Learning Params
NUM_EPOCHS_Q = 500

# UCRL - MNL Parameters
NUM_EPOCHS_UCRL = 100
REG_PARAM = 0.2
CONF_RAD = 1

# File Settings
POLICY_OUTPUT_FILE = './Policy.txt'
PROB_OUPUT_FILE = './Prob.txt'
TD_MC_PLOT = 'td_mc_plot.png'
WRITE_FILE = 'w'
READ_FILE = 'r'

BAYES_MODE = True
NUM_INTERACTIONS = 3


# More states, a bigger state space with a more interesting cases
# Distributional Risk averse algorithms
# Always compare to MC