# Challenge 1 — DQN Agent for Montezuma's Revenge

**Machine Learning Course — Universidad Distrital Francisco José de Caldas**  
**Group 1:** Johan Sebastián Gutiérrez Pérez · Nicolás David Murillo Guerrero

---

## Video

> Link will be added before final submission.


## Environment

- Python 3.10+
- OS: Windows / Linux

---

## Setup and Installation
```bash
# 1. Clone the repository
git clone https://github.com/Johan044/challenge1-dqn-atari
cd challenge1__1

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Install dependencies
pip install .
```

---

## Reproduce the Best Run

The best configuration is `exp_07_more_exploration` (highest episode length, most stable loss).
```bash
python train.py --mode train \
  --model-path models/exp_07_more_exploration \
  --exp-name exp_07_more_exploration \
  --timesteps 200000 \
  --seed 42
```

**Windows PowerShell:**
```powershell
python train.py --mode train --model-path models/exp_07_more_exploration --exp-name exp_07_more_exploration --timesteps 200000 --seed 42
```

---

## Run Full Hyperparameter Sweep (all 8 experiments)
```bash
python train.py --mode sweep --seed 42
```

This reads `sweep_configs.json` and trains all eight configurations sequentially.

---

## Watch the Trained Agent Play
```bash
python train.py --mode play --model-path models/exp_07_more_exploration --episodes 3
```

---

## TensorBoard Metrics
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

Sample logs for all eight experiments are included in the `logs/` folder.

---

## Project Structure