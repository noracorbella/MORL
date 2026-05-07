import mo_gymnasium as mo_gym

envs = [
    'deep-sea-treasure-v0',
    'deep-sea-treasure-concave-v0',
    'deep-sea-treasure-mirrored-v0',
]

for name in envs:
    env = mo_gym.make(name)
    sea_map = env.unwrapped.sea_map

    treasures = [
        (sea_map[row, col])
        for row in range(sea_map.shape[0])
        for col in range(sea_map.shape[1])
        if sea_map[row, col] > 0
    ]

    print(f"\n{name}  {treasures}")
    env.close()