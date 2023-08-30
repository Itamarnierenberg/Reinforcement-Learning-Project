<h2 align="center">Optimizing The Hazard Ratio
</h2> 
<p align="center">
<img src="images/ProjectHeader.png" width="800" height="300" />
  </p>
<h4 align="center">
    Itamar Nierenberg:
  
  <a href="https://www.linkedin.com/in/itamar-nierenberg-1406261b9/"><img src="./images/LinkedInLogo.png" width="120" height="40"/></a>
    <a href="https://github.com/itamarnierenberg"><img src="./images/GitHubLogo.png" width="40" height="40"/></a>
</a>

<h4 align="center">
    Stav Beno:
  
  <a href="https://www.linkedin.com/in/stav-beno-26033a232/"><img src="./images/LinkedInLogo.png" width="120" height="40"/></a>
    <a href="https://github.com/stavbeno/"><img src="./images/GitHubLogo.png" width="40" height="40"/></a>
</a>

This project's goal is to optimize a policy for the Hazard Ratio metric using Advanced RL and Bayesian methods

Table Of Contents
--
* [Background](#background)
* [File Tree](#file-tree)
* [Refrences](#refrences)
* [Environment](#environment)

## Background
In medical research there is a common problem of learning a policy to act with during a medical trial such that we maximize the patients probability to improve their initial condition throughout all the trial in relative to a control group.
The mathematical formulation of this risk is the Hazard Ratio.
Optimizing this metric using little data and realistic assumptions can prove useful in the medical field along with many other possible implementations.

## File Tree
This repository consists of two major folders:
* Environment - Our own implementation of the envrionment without the use of external packages
* DQN - Our in environment in gymnasium, using DQN to learn a policy

### Environment

| File Name                           | Purpose                                              |
|-------------------------------------|------------------------------------------------------|
| `Environment/CategoricalTD.py`      | This file implements the Distributional TD Learning  |
| `Environment/MonteCarlo.py`         | Implements Monte Carlo value distribution estimation |
| `Environment/QLearning.py`          | Q-Learning distributional version                    |
| `Environment/HazardEnv.py`          | This is the class implementing the environment       |
| `Environment/Params.py`             | Configuration file                                   |
| `Environment/PolicyOptimization.py` | Vanilla optimization algorithms (Policy Iteration)   |
| `Environment/runTest.py`            | Main file used to run experiments                    |
| `Environment/Utils.py`              | Utility functions                                    |

### DQN

| File Name                   | Purpose                                               |
|-----------------------------|-------------------------------------------------------|
| `DQN/DQN.py`                | The DQN class                                         |
| `DQN/Patients.py`           | Gymnasium implementation of the environment           |
| `DQN/ReplayMemory.py`       | Used for training                                     |
| `DQN/TestDQN.py`            | This file contains the training functions for the DQN |
| `DQN/Params.py`             | Configuration file                                    |
| `DQNs/Utils.py`             | Utility functions                                     |

## Environment

Each patient consists of N measurements, each measurement assign with a lower bound, upper bound and a starting state.

* States = A tuple where the i'th element is the i'th measurement current value
* Actions = Control or Treatment X Dosage
* Transition Model = Known and constant for the Control Group, unknown and estimated using Bayesian methods for the treatment group

## Testing

We start by simulating a control group consisting of N patients.
Then randomize probabilities for the current treatment, at each fixed number of steps for each patient we will stop and perform policy optimzations
with respect to the most recent data we have gathered in the previous steps. At each step the probability estimations will also update.