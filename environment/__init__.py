import gymnasium as gym

gym.register(
    id="ContinuousPsoGymEnv-v0",
    entry_point="environment.gym_env_continuous:ContinuousPsoGymEnv",
)

gym.register(
    id="DiscretePsoGymEnv-v0",
    entry_point="environment.gym_env_discrete:DiscretePsoGymEnv",
)
