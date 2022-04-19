# RUDDER
Paper implementation: Reinforcement Learning with Delayed Rewards (https://arxiv.org/abs/1806.07857).

The credit assignment problem is one of the major challenges in reinforcement learning. When the reward is sparse, it becomes difficult to know which actions have contributed the most to the subsequent, delayed reward and which actions have caused a decrease in the reward. RUDDER is a novel reinforcement learning approach to tackle the problem of delayd rewards in Markov Decision Processes (MDP).

This repository is a simple implementation of the above paper demonstrating how a one-dimensional agent moving on the number line, which receives a reward only at the end of the episode, can learn the optimum state-action value function without reward shaping. More information can be found at https://ml-jku.github.io/rudder/.
