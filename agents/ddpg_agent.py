from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf

from environment.gym_env_continuous import ContinuousPsoGymEnv
from tf_agents.environments import tf_py_environment

import gymnasium as gym
from agents.utils.experience_buffer import ExperienceBufferStandard as StandardReplayBuffer
from agents.utils.experience_buffer import ExperienceBufferPriority as PriorityReplayBuffer
from agents.model_networks.ddpg_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ContinuousActionsResultsLogger as ResultsLogger
from agents.utils.policy import OrnsteinUhlenbeckActionNoisePolicyWithDecayScaling, NoNoisePolicy


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super(DDPGAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)

        self.actor_network = ActorNetworkModel(config)
        self.actor_network_target = ActorNetworkModel(config)

        self.critic_network = CriticNetworkModel(config)
        self.critic_network_target = CriticNetworkModel(config)
        self.policy = OrnsteinUhlenbeckActionNoisePolicyWithDecayScaling(config)
        self.test_policy = NoNoisePolicy(config)

        self.update_model_target_weights(tau=1.0)
        self.replay_buffer = PriorityReplayBuffer(config=config) if config.use_priority_replay else StandardReplayBuffer(config=config)

    def get_q_values(self, state):
        return self.actor_network.get_action_q_values(state)

    def update_model_target_weights(self, tau=None):
        if tau is None:
            tau = self.config.tau

        if not self.config.use_mock_data:
            # Get the weights of the actor and critic networks
            theta_a, theta_c, theta_a_targ, theta_c_targ = self.actor_network.model.get_weights(), self.critic_network.model.get_weights(), self.actor_network_target.model.get_weights(), self.critic_network_target.model.get_weights()

            # mixing factor tau : we gradually shift the weights...
            theta_a_targ = [theta_a[i] * tau + theta_a_targ[i] * (1 - tau) for i in range(len(theta_a))]
            theta_c_targ = [theta_c[i] * tau + theta_c_targ[i] * (1 - tau) for i in range(len(theta_c))]

            self.actor_network_target.model.set_weights(theta_a_targ)
            self.critic_network_target.model.set_weights(theta_c_targ)

        return False

    def replay_experience(self):
        # if self.replay_buffer.size() < self.config.batch_size * 50:
        if self.is_in_exploration_state:
            return None, None, None  # Not enough experience to replay yet.

        actor_losses = []
        critic_losses = []
        total_losses = []
        if not self.config.use_mock_data:
            for _ in range(self.config.replay_experience_length):
                ISWeights = 1.0
                tree_idx = None
                if self.config.use_priority_replay:
                    tree_idx, samples, ISWeights = self.replay_buffer.sample(self.config.batch_size)
                    states, actions, rewards, next_states, dones = map(lambda x: np.array(x, dtype=np.float32), zip(*samples))
                    states = np.squeeze(states, axis=1)  # shape becomes (batch_size, 150)
                    next_states = np.squeeze(next_states, axis=1)
                else:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)


                next_actions = self.actor_network_target.predict(next_states)
                next_q_values = self.critic_network_target.predict([next_states, next_actions])
                rewards = rewards[:, np.newaxis]  # Shape (64, 1)
                dones = dones[:, np.newaxis]  # Shape (64, 1)

                # Use Bellman Equation. (recursive definition of q-values)
                q_values_targets = rewards + (1 - dones) * self.config.gamma * next_q_values
                q_values_targets = q_values_targets.astype(np.float32)

                # ---------------------------- update critic ---------------------------- #
                critic_loss, td_error = self.critic_network.train(states, actions, q_values_targets, ISWeights)

                # update priority buffer
                if self.config.use_priority_replay:
                    abs_errors = tf.reduce_sum(tf.abs(td_error), axis=1)
                    self.replay_buffer.batch_update(tree_idx, abs_errors)

                # ---------------------------- update actor ---------------------------- #
                actor_loss = self.actor_network.train(states, self.critic_network.model)

                total_losses.append(actor_loss + critic_loss)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        return total_losses, actor_losses, critic_losses

    def save_models(self, step):
        self.actor_network.save_model(step)
        self.critic_network.save_model(step)

    def load_models(self):
        self.actor_network.load_model()
        self.critic_network.load_model()
        self.actor_network_target.load_model()
        self.critic_network_target.load_model()