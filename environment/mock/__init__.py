import gymnasium as gym

gym.register(
    id="MockContinuousPsoGymEnv-v0",
    entry_point="environment.mock.mock_gym_env_continuous:MockContinuousPsoGymEnv",
)

gym.register(
    id="MockDiscretePsoGymEnv-v0",
    entry_point="environment.mock.mock_gym_env_discrete:MockDiscretePsoGymEnv",
)
