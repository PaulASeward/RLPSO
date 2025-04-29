import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import Model
from keras.layers import Input, Dense, BatchNormalization, Lambda, Concatenate, Attention, LayerNormalization
from agents.model_networks.base_model import BaseModel
from tensorflow.keras.optimizers import Adam


class LogStdLayer(tf.keras.layers.Layer):
    def __init__(self, action_dim):
        super(LogStdLayer, self).__init__()
        self.action_dim = action_dim

    def build(self, input_shape):
        # Initializes a single trainable vector of shape [action_dim]
        self.log_std = self.add_weight(
            name='log_std',
            shape=(self.action_dim,),
            initializer=tf.constant_initializer(-0.5),
            trainable=True
        )
        super(LogStdLayer, self).build(input_shape)

    def call(self, inputs):
        # Expand and broadcast log_std to match the batch dimension
        batch_size = tf.shape(inputs)[0]
        log_std_expanded = tf.expand_dims(self.log_std, axis=0)        # shape [1, action_dim]
        log_std_tiled = tf.tile(log_std_expanded, [batch_size, 1])     # shape [batch_size, action_dim]
        return log_std_tiled


class ActorNetworkModel(BaseModel):
    def __init__(self, config):
        super(ActorNetworkModel, self).__init__(config, "actor")

    def nn_model(self):
        initial_input = Input(shape=self.config.state_shape, dtype=tf.float32)

        if self.config.use_attention_layer:
            attention_output = Attention(use_scale=True)([initial_input, initial_input])
            first_input = LayerNormalization()(attention_output)
        else:
            first_input = initial_input

        x = Dense(self.config.actor_layers[0], name="L0", activation='tanh')(first_input)
        for index in range(1, len(self.config.actor_layers)):
            x = Dense(self.config.actor_layers[index], name=f"L{index}", activation='tanh')(x)
            # x = BatchNormalization()(x)

        unscaled_output = Dense(self.config.action_dimensions, name="Output", activation=tf.nn.tanh)(x)
        scaling_factor = (self.config.upper_bound - self.config.lower_bound) / 2.0
        shift_factor = (self.config.upper_bound + self.config.lower_bound) / 2.0
        output = Lambda(lambda x: x * scaling_factor + shift_factor)(unscaled_output)

        # For continuous actions, output a mean plus log_std
        # output = Dense(self.config.action_dimensions, activation=None)(output)
        log_std = LogStdLayer(self.config.action_dimensions)(x)
        # log_std = tf.Variable(initial_value=-0.5 * tf.ones(self.config.action_dimensions, dtype=tf.float32),
        #                       trainable=True, name="log_std")

        model = Model(inputs=initial_input, outputs=[output, log_std])
        self.optimizer = Adam(learning_rate=self.config.actor_learning_rate)

        return model

    def sample_action(self, obs):
        """
        Samples an action from the current policy (Gaussian).
        Returns the action and log-prob of that action under this policy.
        """
        mean, log_std = self.model(obs, training=False)
        std = tf.exp(log_std)
        pi_distribution = tf.random.normal(shape=tf.shape(mean)) * std + mean
        logp = self.gaussian_likelihood(pi_distribution, mean, log_std)

        avg_std = tf.reduce_mean(std)

        return pi_distribution, logp, avg_std

    @staticmethod
    def gaussian_likelihood(x, mu, log_std):
        """
        Compute log probability of x under Diagonal Gaussian with mean mu and log-std log_std.
        """
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2.0 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @tf.function
    def train(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        """
        The PPO clipping objective for the policy.
        """
        with tf.GradientTape() as tape:
            mean, log_std = self.model(observation_buffer, training=True)
            logp = self.gaussian_likelihood(action_buffer, mean, log_std)
            ratio = tf.exp(logp - logprobability_buffer)

            # PPO objective
            min_advantage = tf.where(
                advantage_buffer >= 0,
                (1 + self.config.clip_ratio) * advantage_buffer,
                (1 - self.config.clip_ratio) * advantage_buffer
            )

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))

            # Approximate KL for early stopping
            approx_kl = tf.reduce_mean(logprobability_buffer - logp)

        grads = tape.gradient(policy_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return policy_loss, approx_kl


class CriticNetworkModel(BaseModel):
    def __init__(self, config):
        super(CriticNetworkModel, self).__init__(config, "critic")

    def nn_model(self):
        initial_input = Input(shape=self.config.state_shape, dtype=tf.float32)

        if self.config.use_attention_layer:
            attention_output = Attention(use_scale=True)([initial_input, initial_input])
            first_input = LayerNormalization()(attention_output)
        else:
            first_input = initial_input

        x = Dense(self.config.critic_layers[0], name="L0", activation='tanh')(first_input)
        for index in range(1, len(self.config.critic_layers)-1):
            x = Dense(self.config.critic_layers[index], name=f"L{index}", activation='tanh')(x)
            # x = BatchNormalization()(x)
        output = Dense(units=self.config.critic_layers[-1], name='Output', activation=None)(x)

        model = Model(inputs=initial_input, outputs=output)
        self.optimizer = Adam(learning_rate=self.config.critic_learning_rate)

        return model

    @tf.function
    def train(self, observation_buffer, return_buffer):
        """
        Simple MSE loss for value function: L = mean( (V(s) - ret)^2 ).
        """
        with tf.GradientTape() as tape:
            value_pred = self.model(observation_buffer, training=True)
            value_loss = tf.reduce_mean((return_buffer - value_pred) ** 2)

        grads = tape.gradient(value_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return value_loss