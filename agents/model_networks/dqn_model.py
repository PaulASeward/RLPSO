import os

import tensorflow as tf
from keras.layers import Input, Dense
from agents.model_networks.base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(self, config):
        super(DQNModel, self).__init__(config, "dqn")

    def nn_model(self):
        model = tf.keras.Sequential(
            [
                Input((self.config.observation_length,)),
                Dense(256, activation="relu"),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.config.num_actions),
            ]
        )

        model.compile(loss=self.compute_loss, optimizer=self.optimizer)

        return model


    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss
