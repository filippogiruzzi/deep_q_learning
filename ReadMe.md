# Deep Q-learning with TensorFlow

Keywords: Python, TensorFlow, Deep Reinforcement Learning, 
Deep Q-learning, Deep Q Networks

## Table of contents

1. [ Installation ](#1-installation)
2. [ Introduction ](#2-introduction)  
    2.1 [ Goal ](#21-goal)  
    2.2 [ Results ](#22-results)  
3. [ Project structure ](#3-project-structure)
4. [ Environments ](#4-environments)
5. [ Project usage ](#5-project-usage)  
    5.1 [ Train a DQN agent to play SpaceInvaders 
    via Reinforcement Learning ](#51-train-a-dqn-agent-to-play-spaceinvaders-via-reinforcement-learning)
6. [ Todo ](#6-todo)
7. [ Resources ](#7-resources)

## 1. Installation

This project was designed for:
* Python 3.6
* TensorFlow 1.12.0

Please install requirements & project:
```
$ cd /path/to/project/
$ git clone https://github.com/filippogiruzzi/deep_q_learning.git
$ cd deep_q_learning/
$ pip3 install -r requirements.txt
$ pip3 install -e . --user --upgrade
```

## 2. Introduction

### 2.1 Goal

The purpose of this project is to design and implement 
Deep Q-learning algorithms such as Deep Q Networks (DQN) or 
Double Deep Q Networks (DDQN) to play games (e.g Atari games) 
with Reinforcement Learning.

### 2.2 Results

## 3. Project structure

The project `deep_q_learning/` has the following structure:
* `dqn/models/`: pre-processing, Q-value & target 
network estimators, DQN agent
* `dqn/training/`: online training
* `dqn/inference/`: AI playing

## 4. Environments

The project supports the OpenAI Gym only for now. More games 
shall be added in the future.

## 5. Project usage

```
$ cd /path/to/project/deep_q_learning/dqn/
```

### 5.1 Train a DQN agent to play SpaceInvaders via Reinforcement Learning

```
$ python3 training/train.py
```

## 6. Todo

- [ ] Finish remodeling
- [ ] Full training on Google colab
- [ ] Add inference scripts
- [ ] Add quantitative results & recordings to ReadMe

## 7. Resources

This project was widely inspired by:

* _Reinforcement Learning tutorials_, 
Denny Britz, 
[ Github ](https://github.com/dennybritz/reinforcement-learning)
* _Reinforcement Learning tutorials_, 
Woongwon, Youngmoo, Hyeokreal, Uiryeong, Keon, 
[ Github ](https://github.com/rlcode/reinforcement-learning)
* _CS 285 (UC Berkeley) - Deep Reinforcement Learning_, 
Sergey Levine,
[ Course page ](http://rail.eecs.berkeley.edu/deeprlcourse/)
* _Personal Reinforcement Learning notes_,
Filippo Giruzzi,
[ Github ](https://github.com/filippogiruzzi/reinforcement_learning_resources)
* _Playing Atari with Deep Reinforcement Learning_, 
Mnih et al., 2013,
[ Arxiv ](https://arxiv.org/abs/1312.5602)
* _Deep Reinforcement Learning to play Space Invaders_, 
Deshai, Banerjee,
[ Project report ](https://nihit.github.io/resources/spaceinvaders.pdf) 