
BODY_TEMP = {'name': 'Body Temperature',
             'min_val': 35.0,
             'max_val': 43.0,
             'res': 0.5,
             'start_state': 40.0}
FEATURES = [BODY_TEMP]
CONTROL_ACTION = 'Dont Take a Pill'
TREATMENT_ACTION = 'Take a Pill'
ACTIONS = [CONTROL_ACTION, TREATMENT_ACTION]
RES = [-0.5, 0, 0.5]
SIZE_OF_CONTROL_GROUP = 2
HORIZON = 100
DISTANCE_FUNC = 'L1'
CONTROL_PROB = [0.25, 0.5, 0.25]
TREATMENT_PROB = [0.4, 0.5, 0.1]

NUM_EPOCHS_TD = 500
STEP_SIZE = 0.1
DISCOUNT_FACTOR = 1
X_AXIS_LOWER_BOUND = -100
X_AXIS_UPPER_BOUND = 100
X_AXIS_RESOLUTION = 100
