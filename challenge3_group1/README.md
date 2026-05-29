# Challenge 3 - PPO for Atari (Montezuma's Revenge)

## Overview

This project implements a Proximal Policy Optimization (PPO) agent for the Atari environment **ALE/MontezumaRevenge-v5** as part of Challenge 3 of the Machine Learning course.

The objective is to compare PPO against the Deep Q-Network (DQN) agent developed in Challenge 1 under equivalent preprocessing, evaluation, and computational budget conditions.

The implementation includes:

* PPO with clipped surrogate objective
* Generalized Advantage Estimation (GAE)
* Shared convolutional actor-critic network
* Entropy regularization
* Gradient clipping
* TensorBoard logging
* Periodic deterministic evaluation
* Hyperparameter sweep support

---

## Environment

Environment:

ALE/MontezumaRevenge-v5

Preprocessing:

* Grayscale observations
* 84 × 84 resize
* Frame skip = 4
* Frame stack = 4
* Observation scaling to [0,1]

---

## Installation

Create a Python virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train a single PPO configuration:

```bash
python train.py
```

Train with custom hyperparameters:

```bash
python train.py --lr 1e-4 --horizon 1024 --ent_coef 0.02 --seed 0
```

Run the predefined hyperparameter sweep:

```bash
python train.py --sweep
```

---

## Evaluation

Evaluate a saved checkpoint:

```bash
python evaluate.py --checkpoint <checkpoint_path>
```

Example:

```bash
python evaluate.py --checkpoint logs/montezuma_ppo/sweep_04_lr0.00025_h1024_ep6_bs64_ent0.02_seed0/checkpoints/best_model.pt
```

---

## Logging

TensorBoard logs are stored in:

```text
logs/montezuma_ppo/
```

Launch TensorBoard:

```bash
tensorboard --logdir logs/montezuma_ppo
```

---

## Project Structure

```text
challenge3_group1/
│
├── train.py
├── evaluate.py
├── model.py
├── ppo.py
├── env_utils.py
├── sweep_configs.json
│
├── logs/
│   └── montezuma_ppo/
│
└── checkpoints/
```

---

## Results Summary

PPO was evaluated on Montezuma's Revenge using multiple random seeds and hyperparameter configurations.

The best observed training performance reached a rolling mean reward of approximately 5.0 during training. However, deterministic evaluation returns remained at 0.0 across all evaluated checkpoints and seeds.

These results suggest that PPO occasionally discovered rewarding trajectories through exploration but was unable to consistently reproduce them during evaluation, highlighting the exploration challenges posed by Montezuma's Revenge.

---

## Authors

Machine Learning Course

Universidad Distrital Francisco José de Caldas
