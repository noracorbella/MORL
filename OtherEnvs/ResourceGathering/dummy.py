import mo_gymnasium as mo_gym
env = mo_gym.make("resource-gathering-v0", enemy_kill_prob=1)
print(env.unwrapped.enemy_kill_prob)