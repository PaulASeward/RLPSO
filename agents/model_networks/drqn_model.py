import tensorflow as tf
from keras.layers import Input, Dense, LSTM
from agents.model_networks.base_model import BaseModel


class DRQNModel(BaseModel):
    def __init__(self, config):
        super(DRQNModel, self).__init__(config, "drqn")

        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.config.trace_length, self.config.observation_length)),
                LSTM(256, activation="tanh"),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.config.num_actions),
            ]
        )

    def train(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)

            assert targets.shape == logits.shape

            loss = self.compute_loss(targets, logits)
            grads = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            loss = loss.numpy()

            return loss