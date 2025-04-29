from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf

from environment.gym_env_continuous import ContinuousPsoGymEnv
from tf_agents.environments import tf_py_environment

import gymnasium as gym
from agents.utils.experience_buffer import PPOBuffer
from agents.model_networks.ppo_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ContinuousActionsResultsLogger as ResultsLogger
from agents.utils.policy import PPOPolicy, NoNoisePolicy

# Currently Deprecated and Has Bugs

class PPOAgent(BaseAgent):
    def __init__(self, config):
        super(PPOAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)

        self.actor_network = ActorNetworkModel(config)
        self.critic_network = CriticNetworkModel(config)
        self.policy = PPOPolicy(config, actor_network=self.actor_network)
        self.test_policy = NoNoisePolicy(config)

        self.current_kl = None
        self.current_value = None
        self.trajectory_buffer = PPOBuffer(config)

    def replay_experience(self):
        """
        Perform the PPO update: fetch the entire on-policy batch from PPOBuffer,
        then run gradient descent for a certain number of epochs on policy and value networks.
        """
        observation_buffer, action_buffer, return_buffer, advantage_buffer, logprobability_buffer = self.trajectory_buffer.get()
        policy_losses, value_losses, kls = [], [], []
        # Update the policy for a certain number of iterations
        for _ in range(self.config.train_policy_iterations):
            policy_loss, kl = self.actor_network.train(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)
            policy_losses.append(policy_loss)
            kls.append(kl)
            if kl > 1.5 * self.config.target_kl:
                # Early stopping if KL grows too big
                break

        # Update the value function for a certain number of iterations
        for _ in range(self.config.train_value_iterations):
            value_loss = self.critic_network.train(observation_buffer, return_buffer)
            value_losses.append(value_loss)

        self.current_kl = np.mean(kls if len(kls) > 0 else [0.0])

        avg_policy_loss, avg_value_loss = np.mean(policy_losses), np.mean(value_losses)
        total_loss = avg_policy_loss + avg_value_loss
        return total_loss, avg_policy_loss, avg_value_loss

    def update_model_target_weights(self):
        self.trajectory_buffer.finish_path(0)

        if self.current_kl > 1.5 * self.config.target_kl:
            print(f"Early stopping due to reaching max KL.")
            print(f"Current KL: {self.current_kl}")
            print(f"Target KL: {self.config.target_kl}")
            print(f"Current KL > 1.5 * Target KL: {self.current_kl > 1.5 * self.config.target_kl}")
            return False
        return False

    def initialize_current_state(self):
        # self.policy.reset()
        observation, swarm_info = self.env.reset()
        return tf.convert_to_tensor(observation.reshape(1, -1), dtype=tf.float32)

    def update_memory_and_state(self, current_state, action, reward, next_observation, terminal):
        # TODO: How do I better access value and logp?
        value = self.current_value
        logp = self.policy.logp

        next_state = tf.convert_to_tensor(next_observation.reshape(1, -1), dtype=tf.float32)
        self.trajectory_buffer.store(current_state, action, reward, value, logp)
        return next_state

    def get_q_values(self, state):
        value = self.critic_network.predict(state)
        self.current_value = value[0]
        return state

    def save_models(self, step):
        self.actor_network.save_model(step)
        self.critic_network.save_model(step)

    def load_models(self):
        self.actor_network.load_model()
        self.critic_network.load_model()
