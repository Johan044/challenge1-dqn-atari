# Challenge 4 - Group 1

## Environment

ALE/MontezumaRevenge-v5

## Repository Structure

```text
challenge4/group1/

collect_demos.py
bc_train.py
discriminator.py
gail.py

demos/
logs/
checkpoints/

README.md
CHECKLIST.md
demos_info.txt
```

## Collect Demonstrations

```bash
python collect_demos.py
```

## Train Behavioral Cloning

```bash
python bc_train.py
```

## Train GAIL

```bash
python gail.py
```

## TensorBoard

```bash
tensorboard --logdir logs
```

## Demonstration Sources

The demonstrations were collected from the best PPO checkpoint obtained during Challenge 3.

Two demonstration datasets were evaluated:

- 5000 state-action pairs
- 50000 state-action pairs

## Algorithms Compared

- DQN (Challenge 1)
- PPO (Challenge 3)
- Behavioral Cloning (Challenge 4)
- GAIL (Challenge 4)

## Evaluation Metrics

- Mean episode reward
- Evaluation return
- Discriminator accuracy
- Discriminator loss

## Seeds

42