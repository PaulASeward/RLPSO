import numpy as np


class OrnsteinUhlenbeckActionNoise(object):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    :param config.mean: the mean of the noise
    :param config.sigma: the scale of the noise
    :param config.theta: the rate of mean reversion
    :param config.dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
            self,
            config,
            size,
            initial_noise=None,
    ):
        self._theta = config.ou_theta
        self._mu = config.ou_mu
        self._sigma = config.ou_sigma
        self._dt = config.ou_dt

        self._size = size
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros(self._size)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._size)

        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._size)


class NormalNoise:
    def __init__(self, config, size):
        self._mu = config.ou_mu
        self._size = size
        self._sigma = config.ou_sigma

    def __call__(self):
        return np.random.normal(scale=self._sigma, size=self._size)

    def reset(self):
        pass

