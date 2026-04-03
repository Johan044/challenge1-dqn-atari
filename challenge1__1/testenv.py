import gymnasium as gym
import ale_py

# registrar los entornos de Atari
gym.register_envs(ale_py)

env = gym.make("ALE/MontezumaRevenge-v5")

obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

print("Environment works!")