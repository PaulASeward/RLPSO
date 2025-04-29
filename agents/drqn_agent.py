from agents.agent import BaseAgent
import copy
import numpy as np
from agents.utils.experience_buffer import ExperienceBufferRecurrent as ReplayBuffer
from agents.model_networks.drqn_model import DRQNModel
from utils.logging_utils import DiscreteActionsResultsLogger as ResultsLogger


class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)
        self.episode_states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DRQNModel(config)
        self.target_model = DRQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer(config=config)

    def update_episode_states(self, next_observation):
        self.episode_states = np.roll(self.episode_states, -1, axis=0)
        self.episode_states[-1] = next_observation

    def initialize_current_state(self):
        self.episode_states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4
        observation, swarm_info = self.env.reset()
        self.update_episode_states(observation)
        return np.reshape(self.episode_states, [1, self.config.trace_length, self.config.observation_length])

    def update_memory_and_state(self, current_state, action, reward, next_observation, terminal, add_to_replay_buffer=True):
        prev_states = copy.deepcopy(self.episode_states)
        self.update_episode_states(next_observation)  # Updates the states array removing oldest when adding newest for sliding window

        if add_to_replay_buffer:
            self.replay_buffer.add([prev_states, action, reward * self.config.gamma, self.episode_states, terminal])

            # Oversample exploration if desired. Could add in only explore for positive rewards
            if self.config.over_sample_exploration is not None and self.config.over_sample_exploration > 0 and self.is_in_exploration_state:
                times_to_oversample = int(self.config.over_sample_exploration)
                for _ in range(times_to_oversample):
                    self.replay_buffer.add([prev_states, action, reward * self.config.gamma, self.episode_states, terminal])
        return np.reshape(self.episode_states, [1, self.config.trace_length, self.config.observation_length])

    def save_models(self, step):
        self.model.save_model(step)

    def load_models(self):
        self.model.load_model()
        self.target_model.load_model()
