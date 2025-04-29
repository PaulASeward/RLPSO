import os
import copy

import numpy as np


class Config(object):
    use_discrete_env = None
    use_mock_data = False
    use_priority_replay = False
    reward_function = "fitness_reward"
    # reward_function = "normalized_total_difference_reward"
    penalty_for_negative_reward = 0
    use_attention_layer = False
    use_ou_noise = False

    # AGENT PARAMETERS
    num_episodes = 20
    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 30
    # observation_length = 151
    observation_length = 152

    train_steps = 20000
    log_interval = 200
    eval_interval = 500
    test_episodes = 5
    num_final_tests = 100

    replay_experience_length = 1

    # EXPERIMENT PARAMETERS
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    best_f_standard_pso = [0,0,0,0,0,-26,-112,-21,-26,0,-85,-120,-200,-2279,-4080,-1,-104,-130,-5,-12,-305,-2513,-4594,-276,0,0,0,0]
    swarm_improvement_pso = [0,0,0,0,2,31,20,1,0,18,2,10,20,118,1300,1,43,141,4,1,1,237,1075,1,0,0,0,0]
    standard_deviations = [0.00e+00, 7.82e+05, 7.06e+07, 4.55e+03, 0.00e+00, 4.34e+00, 1.74e+01, 5.51e-2, 1.95e+00, 5.53e-2, 1.51e+01, 1.72e+01, 2.21e+01, 3.80e+02, 6.25e+02, 3.51e-1, 1.55e+01, 2.68e+01, 1.30e+00, 5.12e-1, 5.30e+01, 5.15e+02, 7.06e+02, 5.64e+00, 7.12e+00, 4.61e+01, 7.43e+01, 2.82e-13]

    # Output files
    results_dir = "results"
    # results_dir = "run_history/20241126/f11_PMSO_DDPG"

    standard_pso_results_dir = "pso/standard_pso_results"
    os.makedirs(results_dir, exist_ok=True)
    swarm_locations_dir = os.path.join(results_dir, "swarm_locations")
    os.makedirs(swarm_locations_dir, exist_ok=True)
    env_swarm_locations_name = "swarm_locations.npy"
    env_swarm_velocities_name = "swarm_velocities.npy"
    env_swarm_best_locations_name = "swarm_best_locations.npy"
    env_swarm_evaluations_name = "swarm_evaluations.npy"
    env_meta_data_name = "meta_data.csv"

    # Model/Checkpoint Files
    checkpoint_dir = os.path.join(results_dir, "saved_session", "model_checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log_dir = os.path.join(results_dir, "saved_session", "logs")
    save_models = True
    save_buffer = False
    load_checkpoint_dir = None

    # EPSILON GREEDY PARAMETERS
    policy = "ExponentialDecayGreedyEpsilon"
    epsilon_start = 1.0
    epsilon_end = 0.01


    # Replay Buffer
    # buffer_size = 10000
    # buffer_size = 20000
    buffer_size = 30000
    # buffer_size = 1000000
    # batch_size = 64
    batch_size = 128
    replay_priority_capacity = 100000
    replay_priority_epsilon = 0.01  # small amount to avoid zero priority
    replay_priority_alpha = 0.7  # [0~1] convert the importance of TD error to priority
    replay_priority_beta = 0.5  # importance-sampling, from initial value increasing to 1
    replay_priority_beta_increment = 0.001
    replay_priority_beta_max_abs_error = 1.0  # clipped abs error

    # DRQN TRAINING PARAMETERS
    trace_length = 20

    # DDPG TRAINING PARAMETERS
    ou_mu = None  # Will be set to zeros of action_dim in update_properties
    ou_theta = 0.15
    ou_sigma = 0.25
    # ou_sigma = 0.5
    ou_dt = 1e-2

    tau = 0.005
    # tau = 0.125
    upper_bound = None
    lower_bound = None
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    actor_layers = (256, 128, 64)
    critic_layers = (256, 128, 64)
    subswarm_action_dim = None
    state_shape = None
    action_bound = None
    action_shift = None

    # PPO TRAINING PARAMETERS
    clip_ratio = 0.2
    target_kl = 0.01
    lam = 0.97

    # policy_learning_rate = 3e-4
    # value_function_learning_rate = 1e-3
    # train_policy_iterations = 80
    # train_value_iterations = 80
    train_policy_iterations = 10
    train_value_iterations = 10

    actions_descriptions = None
    practical_action_low_limit = None
    practical_action_high_limit = None

    # LEARNING PARAMETERS
    # discount_factor = 0.01
    gamma = 0.99
    learning_rate = 0.001
    lr_method = "adam"

    # learning_rate = 0.00025
    # learning_rate_minimum = 0.00025
    # lr_decay = 0.97
    # keep_prob = 0.8

    # LSTM PARAMETERS
    num_lstm_layers = 1
    lstm_size = 512
    # min_history = 4

    # EVALUATION PARAMETERS
    # number_evaluations = 10000

    # PSO Config:
    topology = 'global'
    is_sub_swarm = False

    w = 0.729844  # Inertia weight
    c1 = 2.05 * w  # Social component Learning Factor
    c2 = 2.05 * w  # Cognitive component Learning Factor

    # w_min = 0.43  # Min of 5 decreases of 10%
    # w_max = 1.175  # Max of 5 increases of 10%
    w_min = 0.23  # Min of 5 decreases of 10%
    w_max = 1.375  # Max of 5 increases of 10%

    c_min = 0.883  # Min of 5 decreases of 10%
    c_max = 2.409  # Max of 5 increases of 10%

    # w_min = 0  # Min of 5 decreases of 10%
    # w_max = 1.44  # Max of 5 increases of 10%
    # c_min = 0  # Min of 5 decreases of 10%
    # c_max = 3.3 # Max of 5 increases of 10%
    rangeF = 100
    # v_min = 59.049
    # v_max = 161.051

    v_min = 50
    v_max = 150

    v_min_scaling_factor = 0.5
    v_max_scaling_factor = 1.5

    velocity_braking = 1.0
    velocity_braking_min = 0.75
    velocity_braking_max = 1.25

    distance_threshold = 0.00
    distance_threshold_min = -0.20
    distance_threshold_max = 0.20

    replacement_threshold = 1.0
    replacement_threshold_min = 0.75
    replacement_threshold_max = 1.25

    def __init__(self):
        self.func_num = None
        self.action_dimensions = None
        self.experiment_config_path = None
        self.action_counts_path = None
        self.continuous_action_history_path = None
        self.action_values_path = None
        self.test_step_results_path = None
        self.action_training_values_path = None
        self.epsilon_values_path = None
        self.fitness_plot_path = None
        self.average_returns_plot_path = None
        self.fitness_path = None
        self.episode_results_path = None
        self.training_step_results_path = None
        self.average_returns_path = None
        self.loss_file = None
        self.actor_loss_file = None
        self.critic_loss_file = None
        self.interval_actions_counts_path = None
        self.standard_pso_path = None
        self.experience_buffer_path = None
        self.experiment = None
        self.num_eval_intervals = None
        self.label_iterations_intervals = None
        self.iteration_intervals = None
        self.obs_per_episode = None
        self.iterations = None
        self.swarm_size = None
        self.num_actions = None
        self.swarm_algorithm = None
        self.num_sub_swarms = None
        self.network_type = None
        self.dim = None
        self.over_sample_exploration = None

    def clone(self):
        return copy.deepcopy(self)

    def update_properties(self, network_type=None, swarm_algorithm=None, func_num=None, num_actions=None, load_checkpoint=None, action_dimensions=None, num_subswarms=None, swarm_size=None, dimensions=None, num_episodes=None, num_swarm_obs_intervals=None, swarm_obs_interval_length=None, train_steps=None, over_sample_exploration=None):
        if func_num is not None:
            self.func_num = func_num

        if num_actions is not None:
            self.num_actions = num_actions

        if load_checkpoint is not None:
            self.load_checkpoint_dir = os.path.join(self.checkpoint_dir, load_checkpoint)

        if dimensions is not None:
            self.dim = dimensions

        if num_episodes is not None:  # The number of episodes in each Reinforcement Learning Iterations before terminating.
            self.num_episodes = num_episodes
            self.trace_length = num_episodes if num_episodes < 20 else 20

        if num_swarm_obs_intervals is not None:  # The number of swarm observation intervals. Ex) At 10 evenly spaced observation intervals, observations in the swarm will be collected. Default is 10.
            self.num_swarm_obs_intervals = num_swarm_obs_intervals

        if swarm_obs_interval_length is not None:  # The number of observations per episode conducted in the swarm. Ex) Particle Best Replacement Counts are averaged over the last _ observations before an episode terminates and action is decided. Default is 30.
            self.swarm_obs_interval_length = swarm_obs_interval_length

        self.obs_per_episode = self.swarm_obs_interval_length * self.num_swarm_obs_intervals

        if train_steps is not None:
            self.train_steps = train_steps
            self.log_interval = train_steps // 100  # normal is 100
            self.eval_interval = train_steps // 40  # Normal is 40
            self.iterations = range(0, train_steps, self.eval_interval)
            self.num_eval_intervals = train_steps // self.eval_interval
            self.iteration_intervals = range(self.eval_interval, train_steps + self.eval_interval, self.eval_interval)
            self.label_iterations_intervals = range(0, train_steps + self.eval_interval, self.train_steps // 20)

        if swarm_algorithm is not None:
            self.swarm_algorithm = swarm_algorithm

        if num_subswarms is not None:
            if self.swarm_algorithm == "PMSO":
                self.num_sub_swarms = num_subswarms
            else:
                self.num_sub_swarms = 1

        if swarm_size is not None:
            self.swarm_size = swarm_size
            # self.observation_length = self.swarm_size * 3 + 1
            self.observation_length = self.swarm_size * 3 + 1 + (1 * self.num_sub_swarms) + (1 * self.num_sub_swarms)

        if action_dimensions is not None:
            if self.num_sub_swarms is not None:
                self.subswarm_action_dim = action_dimensions
                self.action_dimensions = action_dimensions * self.num_sub_swarms
            else:
                self.action_dimensions = action_dimensions
            self.ou_mu = np.zeros(self.action_dimensions)

        if over_sample_exploration is not None:
            self.over_sample_exploration = over_sample_exploration

        if network_type is not None:
            if network_type in ["DQN", "DRQN"]:
                self.use_discrete_env = True
            else:
                self.use_discrete_env = False

            self.network_type = network_type
            experiment = self.network_type + "_" + self.swarm_algorithm + "_F" + str(self.func_num)
            self.experiment = experiment
            self.experiment_config_path = os.path.join(self.results_dir, f"experiment_config.json")
            self.interval_actions_counts_path = os.path.join(self.results_dir, f"interval_actions_counts.csv")
            self.loss_file = os.path.join(self.results_dir, f"average_training_loss.csv")
            self.actor_loss_file = os.path.join(self.results_dir, f"actor_loss.csv")
            self.critic_loss_file = os.path.join(self.results_dir, f"critic_loss.csv")
            self.average_returns_path = os.path.join(self.results_dir, f"average_returns.csv")
            self.fitness_path = os.path.join(self.results_dir, f"average_fitness.csv")
            self.episode_results_path = os.path.join(self.results_dir, f"episode_results.csv")
            self.training_step_results_path = os.path.join(self.results_dir, f"step_results.csv")
            self.test_step_results_path = os.path.join(self.results_dir, f"test_results.csv")
            self.action_values_path = os.path.join(self.results_dir, f"actions_values.csv")
            self.action_training_values_path = os.path.join(self.results_dir, f"actions_training_values.csv")
            self.continuous_action_history_path = os.path.join(self.results_dir, f"continuous_action_history.npy")
            self.action_counts_path = os.path.join(self.results_dir, f"actions_counts.csv")
            self.epsilon_values_path = os.path.join(self.results_dir, f"epsilon_values.csv")
            self.standard_pso_path = os.path.join(self.standard_pso_results_dir, f"f{self.func_num}.csv")
            self.experience_buffer_path = os.path.join(self.results_dir, f"experience_buffer.pkl")