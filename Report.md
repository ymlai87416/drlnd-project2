[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

[image2]: images/ddpg_result.png "DDPG result"

# Project 2: Continuous Control 

## Introduction

For this project, An agent is trained to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

## Environment

Agent is trained in the following environment

* Python 3
* Pytorch 0.4.1 cuda version 9.2
* Ubuntu 18.04
* AMD Ryzen Threadripper 2950X 16-Core Processor
* Nvidia RTX 2800Ti

## Agent

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between `-1` and `1`.

For this project, we will provide you with two separate versions of the Unity environment:

The first version contains a single agent.

The second version contains 20 identical agents, each with its own copy of the environment.


```
Size of each action: 4
There are 20 agents. Each observes a state with length: 33
The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
```

## Algorithm

DDPG [1] is used to train the agent. 

### Actor-Critic

In file `model.py`, the class `Actor` implements the actor network, and the class `Critic` implements the critic network.

Agent create 4 networks, they are `actor_local`, `actor_target`, `critic_local` and `critic_target`. Learning happens on `actor_local` and `critic_local`, and soft update is applied to gradually apply the weighting of the local network to the target network.

### Actor network

The network acts as a policy base method, the input is current state, the output is an optimal action.

```
Actor(
  (fc1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=4, bias=True)
)
```

### Critic network

The critic network as a value base network, input is the current state (as input of fcs1 layer) and the action taken (as input of fc2 layer), output is the expected reward.

```
Critic(
  (fcs1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)
```

Learning function is implemented in file `ddpg_agent.py`. In class `Agent`  function `learn`. The critic network is updated first, then to the actor network. Soft update are then applied to both actor and critic network.


### Reply Buffer

In file `ddpg_agent.py`, the class `ReplayBuffer` implements the replay buffer, the replay buffer save all experience experienced by 20 agents. Random experiences are retrieved from time to time to train both actor and critic network.


### Ornstein-Uhlenbeck Process for Noise generation

In file `ddpg_agent.py`, the class `OUNoise` implements the noise generation process to allow DDPG to explore around the optimal action given by the actor network. Please see the hyperparameter section for parameter value selection.

### Fine tuning

In this implementation, several techniques are employed.

Batch normalization is used to allow learn effectively across different types of units. 

Gradient clipping is also used to keep the score stable across 100 episodes by avoiding the network making updates that are too far away from the current weighting.

Weight decay = 0.01 of Adam optimizer is not adopted although it is mentioned in paper [1]. It is because it stop the agent from learning.

```
Episode 10	Average Score: 0.34
Episode 20	Average Score: 0.62
Episode 30	Average Score: 0.60
Episode 40	Average Score: 0.77
...
```

### Hyperparameters

Here is the network structure of actor 

```
Actor(
  (fc1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=4, bias=True)
)
```

and critic

```
Critic(
  (fcs1): Linear(in_features=33, out_features=400, bias=True)
  (bn1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)
```

Discount factor is of 𝛾 = 0.95

Training occurs at every 20 timesteps and updates the network for 150 times. Mini batch size = 128 experiences.

Actor and critic network initialization:

Final layer: uniform between [-3e-3, 3e-3]

Other layer: uniform between [-1/sqrt(f), 1/sqrt(f)] where f is the fan-in of the layer.

the learning rate for actor is 10<sup>-4</sup> and critic is 10<sup>-3</sup>.

Soft update 𝜏 = 0.001

No weight decay on neither actor nor critic network

Ornstein-Uhlenbeck process: μ = 0, 𝜃 = 0.15, 𝜎 = 0.08



## Result

The graph shows the agent can receive an average reward (over 100 episodes, and overall 200 agents) of at least +30.

![Reward plot][image2]

The agent starts to obtain a score over 30 after episode 40 until the end, which is episode 150.

```
Episode 10	Average Score: 3.37
Episode 20	Average Score: 10.82
Episode 30	Average Score: 23.89
Episode 40	Average Score: 33.56
Episode 50	Average Score: 36.64
Episode 60	Average Score: 38.04
Episode 70	Average Score: 38.55
Episode 80	Average Score: 38.65
Episode 90	Average Score: 37.93
Episode 100	Average Score: 38.02
Episode 110	Average Score: 38.66
Episode 120	Average Score: 38.23
Episode 130	Average Score: 38.30
Episode 140	Average Score: 37.95
Episode 150	Average Score: 38.23
```

You can also watch how the agent perform at [here](https://youtu.be/XWjGVSZObck)

## Future works

Here is some idea to further improve the performance and the learning efficiency of the agent.

* PPO [2] from OpenAI in 2017
* A3C [3] from Google DeepMind in 2016
* D4PG [4] from Google DeepMind in 2018

## Reference

[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015). [link](https://arxiv.org/pdf/1509.02971)

[2] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).  [link](https://arxiv.org/pdf/1707.06347.pdf)

[3] Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. 2016.  [link](https://arxiv.org/pdf/1602.01783.pdf)

[4] Barth-Maron, Gabriel, et al. "Distributed distributional deterministic policy gradients." arXiv preprint arXiv:1804.08617 (2018).  [link](https://arxiv.org/pdf/1804.08617)
