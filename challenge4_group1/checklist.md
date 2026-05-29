# Challenge 4 Checklist

## Demonstration Collection

```bash
python collect_demos.py
```

## Behavioral Cloning

```bash
python bc_train.py
```

## GAIL Training

```bash
python gail.py
```

## Seeds

- 42

## Logs

### DQN

```text
challenge1/group1/logs/
```

### PPO

```text
challenge3_group1/logs/montezuma_ppo/
```

### BC

```text
challenge4/group1/logs/bc/
```

### GAIL 5000

```text
challenge4/group1/logs/gail_5000/
```

### GAIL 50000

```text
challenge4/group1/logs/gail_50000/
```

## Figures

TensorBoard figures were generated for:

- train/mean_reward_100ep
- eval/mean_return
- gail/discriminator_accuracy
- gail/discriminator_loss

## Results Summary

| Algorithm | Final Score |
|------------|------------|
| DQN | 0.0 |
| PPO | 5.0 |
| BC | 1.3 |
| GAIL (5000 demos) | 2.6 |
| GAIL (50000 demos) | 5.8 |

## Comparative Summary

Montezuma's Revenge is an extremely sparse-reward Atari environment. The DQN agent from Challenge 1 was unable to obtain meaningful rewards and remained at zero performance throughout training. PPO improved exploration and achieved a final score close to 5.0, demonstrating that policy-gradient methods can partially overcome the sparse-reward problem. Behavioral Cloning learned useful action patterns directly from demonstrations and reached approximately 1.3 reward, outperforming a random policy but remaining limited by distribution shift. GAIL successfully leveraged expert demonstrations through adversarial imitation learning. With only 5000 demonstrations, GAIL achieved a score around 2.6, showing clear improvement over BC. Increasing the dataset to 50000 demonstrations substantially improved performance, allowing GAIL to reach approximately 5 reward and surpass PPO. The discriminator accuracy gradually approached 50%, indicating that the generated trajectories became increasingly difficult to distinguish from expert demonstrations. These results suggest that imitation learning provides significant advantages in sparse-reward environments where exploration is difficult. The quantity of demonstrations strongly influenced performance, while adversarial reward learning allowed GAIL to outperform pure supervised imitation.