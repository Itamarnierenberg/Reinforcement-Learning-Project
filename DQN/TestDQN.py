# from Patients import Patients
from Utils import *
# import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from ReplayMemory import ReplayMemory
from ReplayMemory import Transition
import math
import random
from DQN import DQN
from itertools import count
from tqdm import tqdm


control_group, death_list = create_control_data()
treat_prob = create_treatment_prob()
print(treat_prob)
patient_list = list()
for patient in tqdm(range(prm.SIZE_OF_ACTION_GROUP), desc='Initializing Environments for all patients in the exp'):
    patient_list.append(Patients(is_control=False, treatment_prob=treat_prob, control_group=control_group))

# env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
print(is_ipython)
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("mps" if torch.has_mps else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = patient_list[0].action_space.n
# Get the number of state observations
n_observations = prm.NUM_FEATURES

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state, patient):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[patient.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_kap_meier(action_deaths, control_deaths, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Time')
    plt.ylabel('# Alive')
    plt.step(list(range(prm.HORIZON)), action_deaths)
    plt.step(list(range(prm.HORIZON)), control_deaths)
    # Take 100 episode averages and plot them too
    #if len(durations_t) >= 100:
    #    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #    means = torch.cat((torch.zeros(99), means))
    #    plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.has_mps:
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in tqdm(range(num_episodes), desc='Training Model'):
    # Initialize the environment and get it's state
    state_list = torch.empty((prm.SIZE_OF_ACTION_GROUP, 1,  prm.NUM_FEATURES), device=device)
    done_list = torch.empty(prm.SIZE_OF_ACTION_GROUP, device=device)
    d_list_exp = np.zeros(prm.HORIZON) + prm.SIZE_OF_ACTION_GROUP
    for idx, patient in enumerate(patient_list):
        state = patient.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state_list[idx] = state
        done_list[idx] = False
    for t in count():
        num_of_deaths = 0
        for idx, patient in enumerate(patient_list):
            if done_list[idx]:
                continue
            action = select_action(state_list[idx], patient)
            observation, reward, terminated, _ = patient.step(action.item())
            reward = np.float32(reward)
            reward = torch.tensor([reward], device=device)
            done_list[idx] = terminated
            if terminated:
                next_state = None
                num_of_deaths += 1
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state_list[idx], action, next_state, reward)

            # Move to the next state
            if not terminated:
                state_list[idx] = next_state
        d_list_exp[t:] -= num_of_deaths

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if all(done_list):
            episode_durations.append(t + 1)
            plot_kap_meier(d_list_exp, death_list, show_result=True)
            # plot_durations()
            break

print('Complete')
# plot_kap_meier(show_result=True)
# plt.ioff()
# plt.show()
