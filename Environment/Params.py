# Features Definition

# BODY_TEMP = {'idx': 0,
#              'min_val': 35.0,
#              'max_val': 43.0,
#              'res': 0.5,
#              'start_state': 40.0}

BODY_TEMP = {'idx': 0,
             'min_val': 39.0,
             'max_val': 41.0,
             'res': 0.5,
             'start_state': 40.0}
FEATURES = [BODY_TEMP]

FEATURES_IDX_TO_NAME = ['Body Temperature', 'Time']

# Actions Definition
CONTROL_ACTION = 0
TREATMENT_ACTION = 1
ACTIONS = [CONTROL_ACTION, TREATMENT_ACTION]
ACTION_IDX_TO_NAME = ["Don't take a pill", "Take a pill"]

RES = [-0.5, 0, 0.5]

# Environment Params
SIZE_OF_CONTROL_GROUP = 100
HORIZON = 2
DISTANCE_FUNC = 'L1'
CONTROL_PROB = [0.25, 0.5, 0.25]
TREATMENT_PROB = [0.3, 0.5, 0.2]
TIME_IDX = -1

# Categorical TD Params
NUM_EPOCHS_TD = 500
STEP_SIZE = 0.1
DISCOUNT_FACTOR = 1
X_AXIS_LOWER_BOUND = -100
X_AXIS_UPPER_BOUND = 100
X_AXIS_RESOLUTION = 1000
