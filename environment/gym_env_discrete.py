import gymnasium as gym
import numpy as np
from pso.cec_benchmark_functions import CEC_functions
from environment.actions.discrete_actions import DiscreteActions, DiscreteMultiswarmActions
from pso.pso_swarm import PSOSwarm
from pso.pso_multiswarm import PSOMultiSwarm


class DiscretePsoGymEnv(gym.Env):
    """Continuous environment."""

    # reward_range = (-float("inf"), float("inf"))

    def __init__(self, config):
        self._func_num = config.func_num
        self._minimum = config.fDeltas[config.func_num - 1]
        self._max_episodes = config.num_episodes
        self._standard_pso_values_path = config.standard_pso_path

        self._best_standard_pso = config.best_f_standard_pso[config.func_num - 1]
        self._pso_variance = config.standard_deviations[config.func_num - 1] ** 2
        self._avg_swarm_improvement = config.swarm_improvement_pso[config.func_num - 1]
        self._avg_standard_pso_increase = self._avg_swarm_improvement / self._max_episodes
        self._penalty_for_negative_reward = config.penalty_for_negative_reward

        self._observation_length = config.observation_length
        low_limits_obs_space = np.zeros(self._observation_length)  # 150-dimensional array with all elements set to 0
        high_limits_obs_space = np.full(self._observation_length, np.inf)


        if config.swarm_algorithm == "PMSO":
            self.swarm = PSOMultiSwarm(objective_function=CEC_functions(dim=config.dim, fun_num=config.func_num), config=config)
            self.actions = DiscreteMultiswarmActions(swarm=self.swarm, config=config)
        else:
            self.swarm = PSOSwarm(objective_function=CEC_functions(dim=config.dim, fun_num=config.func_num), config=config)
            self.actions = DiscreteActions(swarm=self.swarm, config=config)
            config.num_actions = len(self.actions.action_names)

        config.actions_descriptions = self.actions.action_names[:config.num_actions]
        self.action_space = gym.spaces.Discrete(config.num_actions)
        self.observation_space = gym.spaces.Box(low=low_limits_obs_space, high=high_limits_obs_space,
                                                shape=(self._observation_length,), dtype=np.float32)

        self._actions_count = 0
        self._current_episode_percent = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []
        self._best_fitness = None
        self._best_relative_fitness_for_plots = None
        self._best_relative_fitness_for_reward = None

        self.total_difference = 0
        self.last_action = 0

        self.reward_functions = {
            "simple_reward": self.simple_reward,
            "fitness_reward": self.fitness_reward,
            "difference_reward": self.difference_reward,
            "total_difference_reward": self.total_difference_reward,
            "normalized_total_difference_reward": self.normalized_total_difference_reward,
            "smoothed_total_difference_reward": self.smoothed_total_difference_reward,
        }
        self.reward_function = config.reward_function

    def simple_reward(self, difference):
        if difference > 0:
            return np.float32(1)
        else:
            return np.float32(self._penalty_for_negative_reward)

    def fitness_reward(self, difference):
        current_best_fitness = self.swarm.get_current_best_fitness()

        if self._best_relative_fitness_for_reward is None:
            reward = self._minimum - current_best_fitness
            self._best_relative_fitness_for_reward = current_best_fitness
        else:
            reward = self._best_relative_fitness_for_reward - current_best_fitness
            self._best_relative_fitness_for_reward = min(self._best_relative_fitness_for_reward, current_best_fitness)

        return max(reward, self._penalty_for_negative_reward)

    def difference_reward(self, difference):
        return max(difference, self._penalty_for_negative_reward)

    def total_difference_reward(self, difference):
        reward = difference * self.total_difference
        if reward > self.total_difference:
            reward = self.total_difference
        return max(reward, self._penalty_for_negative_reward)

    def smoothed_total_difference_reward(self, difference):
        difference = max(difference, 0)
        if self._current_episode_percent == 0:
            standard_start_value = np.genfromtxt(self._standard_pso_values_path, delimiter=',', skip_header=1)[0,1]
            reward = (standard_start_value - self._best_fitness) * self._avg_standard_pso_increase
            # reward = (standard_start_value - self._best_fitness) / self._pso_variance
            return reward
        else:
            log_difference = np.log(1 + ((difference * self._avg_standard_pso_increase) / ((max(0.05, 1-self._current_episode_percent))**2)))

            # Calculate the reward
            reward = log_difference * self.total_difference

            # Clip the reward to the total difference
            if reward > self.total_difference:
                reward = self.total_difference
            reward /= self._pso_variance

            return max(reward, self._penalty_for_negative_reward)

    def normalized_total_difference_reward(self, difference):
        reward = difference * self.total_difference

        # Clip the reward to the total difference
        if reward > self.total_difference:
            reward = self.total_difference

        reward /= self._pso_variance

        return max(reward, self._penalty_for_negative_reward)

    def _get_obs(self):
        # return self.swarm.get_observation()
        swarm_observation = self.swarm.get_observation()
        observation = np.append(swarm_observation, self._current_episode_percent)
        observation = np.append(observation, self.last_action)

        return observation.astype(np.float32)

    def _get_reward(self):
        current_best_f = self.swarm.get_current_best_fitness()

        difference = self._best_fitness - current_best_f
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_f)

        reward = self.reward_functions[self.reward_function](difference)
        if type(reward) is int:
            reward = np.float32(reward)
        else:
            reward = reward.astype(np.float32)
        return reward

    def _get_fitness_reward_for_plots(self):
        current_best_fitness = self.swarm.get_current_best_fitness()

        if self._best_relative_fitness_for_plots is None:
            reward = self._minimum - current_best_fitness
            self._best_relative_fitness_for_plots = current_best_fitness
        else:
            reward = max(self._best_relative_fitness_for_plots - current_best_fitness, 0)  # no penalty in reward
            self._best_relative_fitness_for_plots = min(self._best_relative_fitness_for_plots, current_best_fitness)

        return reward

    # def _get_reward(self):
    #     current_best_f = self.swarm.get_current_best_fitness()
    #
    #     if self._best_fitness is None:
    #         reward = self._minimum - current_best_f
    #         self._best_fitness = current_best_f
    #     else:
    #         reward = max(self._best_fitness - current_best_f, 0)  # no penalty in reward
    #         self._best_fitness = min(self._best_fitness, current_best_f)
    #
    #     return reward

    def _get_done(self):
        return self._episode_ended

    def _get_info(self):
        info = self.swarm.get_swarm_observation()
        info["fitness_reward"] = self._get_fitness_reward_for_plots()
        return info

    def reset(self, seed=None, return_info=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.last_action = 0
        self._actions_count = 0
        self._episode_ended = False
        self.total_difference = 0
        self._best_relative_fitness_for_plots = None
        self._best_relative_fitness_for_reward = None

        # Restart the swarm with initializing criteria
        self.swarm.reinitialize()
        self._best_fitness = self.swarm.get_current_best_fitness()

        observation = self._get_obs()
        info = self._get_info()
        self._best_relative_fitness_for_plots = None  # Reset again since get_info() uses it

        return observation, info

    def step(self, action):
        if self._episode_ended:
            # Last action ended the episode, so we need to create a new episode:
            print("Terminal episode reached unexpectedly. Please check the environment implementation")
            return self.reset()


        self._actions_count += 1
        self._current_episode_percent = self._actions_count / self._max_episodes
        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        # Implementation of the action
        self.actions(action)
        self.swarm.optimize()
        if not isinstance(action, int):
            action = action.item()

        self.last_action = action
        observation = self._get_obs()
        reward = self._get_reward()
        # truncated = False

        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, info

    def render(self, mode="human"):
        pass
