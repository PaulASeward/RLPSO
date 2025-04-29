from agents.agent import BaseAgent
import numpy as np
from agents.utils.experience_buffer import ExperienceBufferStandard as ReplayBuffer
from agents.model_networks.dqn_model import DQNModel
from utils.logging_utils import DiscreteActionsResultsLogger as ResultsLogger


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)

        self.model = DQNModel(config)
        self.target_model = DQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer(config=config)

    def save_models(self, step):
        self.model.save_model(step)

    def load_models(self):
        self.model.load_model()
        self.target_model.load_model()
