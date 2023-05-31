def categorical_ql(locations, initial_prob, step_size=0.1, num_epochs=500, discount_factor=0.99):
    q_table = np.array((s))
    for epoch in tqdm(range(num_epochs)):
        env.reset()
        curr_state = env.get_state()
        is_terminal = False
        while not is_terminal:
            action = find_action((ql_table,4)
            next_state, reward, is_terminal = env.step(action)
            p_list = np.zeros(X_AXIS_RESOLUTION)
            for j in range(X_AXIS_RESOLUTION):
                if is_terminal:
                    g = reward
                else:
                    g = reward + discount_factor * locations[j]
                if g <= locations[0]:
                    p_list[0] += td_est_prob[next_state][j]
                elif g >= locations[X_AXIS_RESOLUTION - 1]:
                    p_list[X_AXIS_RESOLUTION - 1] += td_est_prob[next_state][j]
                else:
                    i_star = 0
                    while locations[i_star + 1] <= g:
                        i_star += 1
                    eta = (g - locations[i_star]) / (locations[i_star + 1] - locations[i_star])
                    #sif eta <=0:
                        #print(f'Eta = {eta}, g = {g}, location[i_star] = {locations[i_star]}, locations[i_star + 1] = {locations[i_star + 1]}')
                    p_list[i_star] += (1 - eta) * td_est_prob[next_state][j]
                    p_list[i_star + 1] += eta * td_est_prob[next_state][j]

            for i in range(X_AXIS_RESOLUTION):
                q_table[curr_state][action][i] = (1 - step_size) * td_est_prob[curr_state][i] + step_size * p_list[i]
            curr_state = next_state
    return td_est_prob


def calculate_expectation (prob):
    exp = 0
    delta = (X_AXIS_UPPER_BOUND - X_AXIS_LOWER_BOUND) / X_AXIS_RESOLUTION
    for i in range(X_AXIS_RESOLUTION):
        exp += (1 - prob[i])*delta

def find_action (ql_table, actions_num):
    actions = []
    max=0
    for i in range (actions_num):
        curr = calculate_expectation(ql_table[i])
        if curr > max:
            actions.clear()
            actions.append(i)
            max = curr
        elif curr ==  max:
            actions.append(i)