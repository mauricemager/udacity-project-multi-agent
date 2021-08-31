# Udacity's multi-agent tennis training report

## Algorithm
This projects implements the popular Multi-Agent Actor-Critic algorithm (_MADDPG_) based on this [paper](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf).
The algorithm is an extension to the single-agent _DDPG_ algorithm for the continues space domain problems. 
The code for this MADDPG implementation is based on the multi-agent physical deception lab exercise of Udacity's deep reinforcement learning nanodegree.


## Hyper-parameters

| Hyper-parameter name | Value   | Detailed meaning                       |
| :---:                |:----:   | :---:                                  |
| batch-size     | 512      | Number of samples trained in each step |
| τ               | 0.1    | Soft update for target network weights |
| update_frequency     | 5       | Update frequency for target network    |
| γ             | 0.99    | Discount factor                        |
| Replay buffer size   | 1e5     | Capacity of the replay buffer          |
| α_actor	             | 5e-4    | Actor network learning rate   |
| α_critic	             | 5e-4    | Actor network learning rate   |
| hidden_knodes           |[128, 64]| Actor and critic network hidden layer knodes |
| critic_weight_decay     | 0.0   | Decay rate for critic network                 |
| μ_noise          | 0.0  | UO noise mean   |
| σ_noise	             | 0.2    | UO noise variance   |
| τ_noise	             | 0.1    | UO noise theta parameter   |
| β	             | 0.4   | Replay memory training parameter   |
| β_decay	             | 5e-4    | Replay memory decay parameter   |
## Results

The problem is solved (average reward > 13.0) in 461 episodes. A plot of the training curve is shown below

![Training Learning curve](data/Learning_curve.png)

below is a example video of a trained agent

![Trained banana agent](data/trained_agent_20.gif)

## Improvements

Future work will address implementation of more advanced DQN techniques such as _Double DQN_, _Prioritized Experience Replay_ or _Dueling DQN_








