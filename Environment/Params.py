
BODY_TEMP = {'name': 'Body Temperature',
             'min_val': 35,
             'max_val': 43,
             'res': 0.5,
             'start_state': 40}
FEATURES = [BODY_TEMP]
CONTROL_ACTION = 'Dont Take a Pill'
TREATMENT_ACTION = 'Take a Pill'
ACTIONS = [CONTROL_ACTION, TREATMENT_ACTION]
SIZE_OF_CONTROL_GROUP = 2
HORIZON = 100

