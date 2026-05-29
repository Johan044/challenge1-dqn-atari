# CHECKLIST - Challenge 3 (Group 1)

## Best PPO Configuration

Training command:

```bash
python train.py --horizon 1024 --n_epochs 6 --batch_size 64 --lr 0.00025 --ent_coef 0.02 --seed 0
```

Best run:

```text
logs/montezuma_ppo/sweep_04_lr0.00025_h1024_ep6_bs64_ent0.02_seed0
```

Checkpoint:

```text
logs/montezuma_ppo/sweep_04_lr0.00025_h1024_ep6_bs64_ent0.02_seed0/checkpoints/best_model.pt
```

---

## Seeds Used

* Seed 0
* Seed 1
* Seed 2

---

## Logs and Figures

PPO logs:

```text
logs/montezuma_ppo/
```

DQN logs:

```text
challenge1/logs/
```

TensorBoard launch:

```bash
tensorboard --logdir logs
```

---

## PPO Evaluation Summary

Evaluated checkpoints:

* sweep_00 (seed 0)
* sweep_01 (seed 1)
* sweep_02 (seed 2)
* sweep_04 (short-horizon ablation)

Observed deterministic evaluation return:

```text
Mean Return = 0.0
Std Return = 0.0
Non-zero Episodes = 0/10
```

for all evaluated checkpoints.

---

##DQN vs PPO Summary

The PPO agent was evaluated on ALE/MontezumaRevenge-v5 using the same preprocessing protocol and computational budget employed for the DQN agent developed in Challenge 1. Multiple PPO configurations and random seeds were tested. During training, PPO occasionally achieved positive rewards, with the best configuration reaching a rolling mean reward of approximately 5.0. However, deterministic evaluation performance remained at zero across all evaluated checkpoints and seeds.

The DQN agent exhibited similar behaviour. Although some reward events were observed during training, the final evaluation return also remained at zero. Therefore, neither algorithm succeeded in learning a robust policy capable of consistently reproducing rewarding trajectories.

These results highlight the extreme exploration difficulty of Montezuma’s Revenge. Rewards are sparse, delayed, and require long sequences of precise actions, making the environment particularly challenging for standard deep reinforcement learning algorithms. PPO’s on-policy updates and DQN’s replay-buffer-based learning were both insufficient to overcome the exploration bottleneck under the available computational budget.

Overall, the results suggest that the dominant limitation was not policy optimization but exploration. More advanced exploration techniques such as curiosity-driven learning, intrinsic motivation, or Go-Explore would likely be required to achieve meaningful performance improvements in this environment.
