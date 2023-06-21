import HazardEnv as HazardEnv
import Params as prm
from Utils import state_to_idx_dict
def policy_evaluation(env, policy, gama = 1, epsilon =0.01) :
    values = np.array(env.get_num_states())
    delta  = 0
    while (true):
        for state in state_to_idx_dict(BODY_TEMP):
            val = 0
            for idx, next_state, next_reward in enumerate(env.get_neighbors()):

                val +=prm.TREATMENT_PROB[idx]*(next_reward + gama * values[next_state])
            delta = np.max(delta, values[idx]-val)
            values[i] = val
        if delta < epsilon :
            return values