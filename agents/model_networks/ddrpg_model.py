import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Input, Dense, LSTM, BatchNormalization, Lambda, Concatenate, Attention, LayerNormalization
from agents.model_networks.base_model import BaseModel
from tensorflow.keras.optimizers import Adam


class ActorNetworkModel(BaseModel):
    def __init__(self, config):
        super(ActorNetworkModel, self).__init__(config, "actor")

    def nn_model(self):
        init = tf.random_normal_initializer(stddev=0.0005)

        # State as input
        initial_input = Input(shape=(self.config.trace_length, *self.config.state_shape), dtype=tf.float32)

        x = LSTM(self.config.lstm_size, return_sequences=False)(initial_input)
        x = BatchNormalization()(x)

        # Dense layers
        for index, layer_size in enumerate(self.config.actor_layers):
            x = Dense(layer_size, name=f"L{index}", activation=tf.nn.leaky_relu, kernel_initializer=init)(x)
            x = BatchNormalization()(x)

        # Output layer
        unscaled_output = Dense(self.config.action_dimensions, name="Output", activation=tf.nn.tanh)(x)
        scaling_factor = (self.config.upper_bound - self.config.lower_bound) / 2.0
        shift_factor = (self.config.upper_bound + self.config.lower_bound) / 2.0
        output = Lambda(lambda x: x * scaling_factor + shift_factor)(unscaled_output)

        model = Model(inputs=initial_input, outputs=output)
        self.optimizer = Adam(learning_rate=self.config.actor_learning_rate)

        return model

    @tf.function
    def train(self, states, critic_network):
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
            q_values = critic_network([states, actions])
            actor_loss = -tf.reduce_mean(q_values)

        actor_grads = tape.gradient(actor_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        return actor_loss


class CriticNetworkModel(BaseModel):
    def __init__(self, config):
        super(CriticNetworkModel, self).__init__(config, "critic")

    def nn_model(self):
        init = tf.random_normal_initializer(stddev=0.0005)

        # State as input
        state_input = Input(shape=(self.config.trace_length, *self.config.state_shape), dtype=tf.float32)
        action_input = Input(shape=(self.config.trace_length, self.config.action_dimensions), dtype=tf.float32)

        state_features = LSTM(self.config.lstm_size, return_sequences=False)(state_input)
        state_features = BatchNormalization()(state_features)
        action_features = LSTM(self.config.lstm_size, return_sequences=False)(action_input)
        action_features = BatchNormalization()(action_features)

        concat = Concatenate(axis=-1)([state_features, action_features])

        x = Dense(self.config.critic_layers[0], name="L0", activation=tf.nn.leaky_relu, kernel_initializer=init)(concat)
        for index in range(1, len(self.config.critic_layers)):
            x = Dense(self.config.critic_layers[index], name=f"L{index}", activation=tf.nn.leaky_relu, kernel_initializer=init)(x)
            x = BatchNormalization()(x)

        output = Dense(1, name="Output", kernel_initializer=init)(x)
        model = Model(inputs=[state_input, action_input], outputs=output)

        self.optimizer = Adam(learning_rate=self.config.critic_learning_rate)
        return model

    @tf.function
    def train(self, states, actions, q_values_targets):
        with tf.GradientTape() as tape:
            q_values = self.model([states, actions])
            td_error = q_values - q_values_targets
            critic_loss = tf.reduce_mean(1.0 * tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grad, self.model.trainable_variables))
        return critic_loss, td_error
