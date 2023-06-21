import HazardEnv as HazardEnv
def policy_evaluation(HazardEnv env, policy):
    env = HazardEnv()
    values = np.array(env.get_num_states())
    for i in range(env.get_num_states()):
        values[i] = env.get
