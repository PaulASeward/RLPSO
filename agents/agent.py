import tensorflow as tf
import gymnasium as gym
import os
from datetime import datetime
from environment.gym_env_discrete import DiscretePsoGymEnv
from agents.utils.policy import ExponentialDecayGreedyEpsilonPolicy, GreedyPolicy
import numpy as np


class BaseAgent:
    def __init__(self, config):
        self.replay_buffer = None
        self.target_model = None
        self.model = None
        self.policy = None
        self.test_policy = None
        self.config = config
        self.log_dir = os.path.join(config.log_dir, config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.results_logger = None
        self.starting_step = 0
        self.is_in_exploration_state = True

        self.raw_env = None
        self.env = None

        self.build_environment()
        if config.policy == "ExponentialDecayGreedyEpsilon":
            self.policy = ExponentialDecayGreedyEpsilonPolicy(epsilon_start=config.epsilon_start, epsilon_end=config.epsilon_end, num_steps=config.train_steps, num_actions=config.num_actions)
            self.test_policy = GreedyPolicy()

    def build_environment(self):
        if self.config.use_discrete_env:
            if self.config.use_mock_data:
                self.raw_env = gym.make("MockDiscretePsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("DiscretePsoGymEnv-v0", config=self.config)
                self.env = self.raw_env

            return self.env

        if self.config.use_mock_data:
            self.raw_env = gym.make("MockContinuousPsoGymEnv-v0", config=self.config)
            self.env = self.raw_env
        else:
            self.raw_env = gym.make("ContinuousPsoGymEnv-v0", config=self.config)
            self.env = self.raw_env

        self.config.state_shape = self.env.observation_space.shape

        return self.env

    def get_q_values(self, state):
        return self.model.get_action_q_values(state)

    def update_model_target_weights(self):
        if not self.config.use_mock_data:
            weights = self.model.model.get_weights()
            self.target_model.model.set_weights(weights)

        return False

    def replay_experience(self):
        # if self.replay_buffer.size() < self.config.batch_size * 50:
        if self.is_in_exploration_state:
            return None, None, None  # Not enough experience to replay yet.

        losses = []
        if not self.config.use_mock_data:
            for _ in range(self.config.replay_experience_length):
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)
                targets = self.target_model.predict(states)

                next_q_values = self.target_model.predict(next_states).max(axis=1)

                q_values_target = rewards + (1 - done) * self.config.gamma * next_q_values
                targets[range(self.config.batch_size), actions] = q_values_target

                loss = self.model.train(states, targets)
                losses.append(loss)

        return losses, None, None

    def get_actions(self):
        print(f"num_actions: {self.config.num_actions}")

        for index, description in enumerate(self.config.actions_descriptions):
            if index + 1 <= self.config.num_actions:
                action_no = str(index + 1)
                print(f"Action #{action_no} Description: {description}")

    def initialize_current_state(self):
        self.policy.reset()
        observation, swarm_info = self.env.reset()
        return np.reshape(observation, (1, self.config.observation_length))

    def update_memory_and_state(self, current_state, action, reward, next_observation, terminal, add_to_replay_buffer=True):
        next_state = np.reshape(next_observation, (1, self.config.observation_length))
        # self.replay_buffer.add([current_state, action, reward*self.config.gamma, next_state, terminal])
        if add_to_replay_buffer:
            self.replay_buffer.add([current_state, action, reward, next_state, terminal])

            # Oversample exploration if desired. Could add in only explore for positive rewards
            if self.config.over_sample_exploration is not None and self.config.over_sample_exploration > 1 and self.is_in_exploration_state:
                times_to_oversample = int(self.config.over_sample_exploration)
                for _ in range(times_to_oversample):
                    self.replay_buffer.add([current_state, action, reward, next_state, terminal])
        return next_state

    def save_models(self, step):
        pass

    def load_from_checkpoint(self, step):
        self.replay_buffer.load()
        self.load_models()
        self.starting_step = step
        episilon_values = np.loadtxt(self.config.epsilon_values_path, delimiter=",")
        self.policy.current_epsilon = episilon_values[-1]
        self.policy.step = step * self.config.num_episodes

    def load_models(self):
        pass

    def train(self):
        with self.writer.as_default():
            for step in range(self.starting_step, self.config.train_steps):
                actions, rewards, fitness_rewards, swarm_observations, terminal = [], [], [], [], False
                current_state = self.initialize_current_state()
                if step > 250:
                    self.is_in_exploration_state = False

                while not terminal:
                    q_values = self.get_q_values(current_state)
                    action = self.policy.select_action(q_values)
                    next_observation, reward, terminal, swarm_info = self.env.step(action)

                    current_state = self.update_memory_and_state(current_state, action, reward, next_observation, terminal)

                    fitness_reward = swarm_info["fitness_reward"]  # This is for plotting swarm improvements, not learning purposes.
                    actions.append(action)
                    fitness_rewards.append(fitness_reward)
                    rewards.append(reward)
                    swarm_observations.append(swarm_info)

                    self.replay_experience()
                    self.update_model_target_weights()

                if step % self.config.eval_interval == 0:
                    # Run a test episode to evaluate the model without noise
                    self.test(step)
                    self.save_models(step)
                    self.replay_buffer.save()

                [losses, actor_losses, critic_losses] = self.replay_experience()
                early_stop = self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.
                self.results_logger.save_log_statements(step=step + 1, actions=actions, fitness_rewards=fitness_rewards, training_rewards=rewards,
                                                        train_loss=losses, epsilon=self.policy.current_epsilon,
                                                        swarm_observations=swarm_observations, actor_losses=actor_losses, critic_losses=critic_losses)

                if early_stop:
                    break
            self.results_logger.print_execution_time()

    def test(self, step=None, number_of_tests=None):
        if self.test_policy is None:
            return

        if step is None:
            step = self.config.train_steps

        if number_of_tests is None:
            number_of_tests = self.config.test_episodes

        cumulative_training_rewards = []
        cumulative_fitness_rewards = []

        for _ in range(number_of_tests):
            actions, rewards, fitness_rewards, swarm_observations, terminal = [], [], [], [], False
            current_state = self.initialize_current_state()

            while not terminal:
                q_values = self.get_q_values(current_state)
                action = self.test_policy.select_action(q_values)
                next_observation, reward, terminal, swarm_info = self.env.step(action)
                current_state = self.update_memory_and_state(current_state, action, reward, next_observation, terminal, add_to_replay_buffer=False)

                fitness_reward = swarm_info[
                    "fitness_reward"]  # This is for plotting swarm improvements, not learning purposes.
                actions.append(action)
                fitness_rewards.append(fitness_reward)
                rewards.append(reward)
                swarm_observations.append(swarm_info)

            cumulative_training_reward = np.sum(rewards)
            cumulative_fitness_reward = np.sum(fitness_rewards)
            print(f"EVALUATION STEP #{step} Fitness Reward:{cumulative_fitness_reward} Training Reward: {cumulative_training_reward}")
            print("EVALUATION_ACTION: ", actions)

            cumulative_fitness_rewards.append(cumulative_fitness_reward)
            cumulative_training_rewards.append(cumulative_training_reward)

        avg_fitness_reward = np.mean(cumulative_fitness_rewards)
        std_dev_fitness_reward = np.std(cumulative_fitness_rewards)
        avg_training_reward = np.mean(cumulative_training_rewards)

        print(f"Average Fitness Reward: {avg_fitness_reward} Average Training Reward: {avg_training_reward}")
        self.results_logger._save_to_csv([step, self.policy.current_epsilon, avg_fitness_reward, std_dev_fitness_reward, avg_training_reward], self.config.test_step_results_path)
