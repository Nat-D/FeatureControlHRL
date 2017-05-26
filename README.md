# Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning

  This repository contains code used in the experiments in our paper. "Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning"
  by Nat Dilokthanakul, Christos Kaplanis, Nick Pawlowski, Murray Shanahan (https://arxiv.org/abs/1705.06769)

  We adapted the code from an open-source implementation of A3C, namely, “Universe-StarterAgent”. (https://github.com/openai/universe-starter-agent)

# Dependencies

* Python 2.7
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

# Getting Started

```
conda create --name universe-starter-agent 
source activate universe-starter-agent

brew install tmux htop cmake      # On Linux use sudo apt-get install -y tmux htop cmake

pip install gym[atari]
pip install universe
pip install six
pip install tensorflow
conda install -y -c https://conda.binstar.org/menpo opencv3
conda install -y numpy
conda install -y scipy
```
Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
```source activate universe-starter-agent```


# Abstract

  The problem of sparse rewards is one of the hardest challenges in contemporary reinforcement learning. Hierarchical reinforcement learning (HRL) tackles this problem by using a set of temporally-extended actions, or options, each of which has its own subgoal. These subgoals are normally handcrafted for specific tasks. Here, though, we introduce a generic class of subgoals with broad applicability in the visual domain. Underlying our approach (in common with work using "auxiliary tasks") is the hypothesis that the ability to control aspects of the environment is an inherently useful skill to have. We incorporate such subgoals in an end-to-end hierarchical reinforcement learning system and test two variants of our algorithm on a number of games from the Atari suite. We highlight the advantage of our approach in one of the hardest games -- Montezuma's revenge -- for which the ability to handle sparse rewards is key. Our agent learns several times faster than the current state-of-the-art HRL agent in this game, reaching a similar level of performance.
  
# Reproducing the results

## Experiment 1: Influence of the meta-controller on performance

![ex1](https://github.com/Nat-D/FeatureControlHRL/blob/master/imgs/fig1.png "Results of experiment 1")

In this experiment, we evaluated the performance of the pixel-control agent (top row) and the feature-control agent (bottom row).
To reproduce the results, checkout to branch pixel_control for pixel-control agent and feature_control for feature-control agent.
For example, use the following command for Montezuma's Revenge:

    python train.py -w 8 -e MontezumaRevenge-v0 -l ~/experiments/montezuma_experiment

To change the value of beta, edit line 136 of a3c.py to the value of beta we want. For example, for beta = 0.75:

    self.beta = 0.75

## Experiment 2, 3: With different backpropagation through time (BPTT) length

![ex2](https://github.com/Nat-D/FeatureControlHRL/blob/master/imgs/fig2.png "Results of experiment 2")
![ex3](https://github.com/Nat-D/FeatureControlHRL/blob/master/imgs/fig3.png "Results of experiment 3")

In this experiment, we improve performance by changing the BPTT length from 20 to 100.
In order to run experiments with BPTT = 100, checkout branch feature_control_bptt100

