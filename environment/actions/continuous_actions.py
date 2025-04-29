import numpy as np
from pso.pso_multiswarm import PSOSubSwarm


class ContinuousMultiswarmActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config
        self.subswarm_actions = [ContinuousActions(sub_swarm, config) for sub_swarm in self.swarm.sub_swarms]
        self.action_names = [
            f"SubSwarm {i + 1} {action_name}"
            for i, subswarm in enumerate(self.subswarm_actions)
            for action_name in subswarm.action_names
        ]

        self.practical_action_high_limit = [
            limit
            for subswarm in self.subswarm_actions
            for limit in subswarm.practical_action_high_limit
        ]
        self.practical_action_low_limit = [
            limit
            for subswarm in self.subswarm_actions
            for limit in subswarm.practical_action_low_limit
        ]

    def __call__(self, action):
        # Restructure the flattened action from size(config.num_sub_swarms * 3) to size (config.num_sub_swarms, 3)
        # reshaped_arr = action.reshape(3, 5)
        reformatted_action = np.array(action).reshape(self.config.num_sub_swarms, self.config.subswarm_action_dim)

        # Action should be a dedicated action for each subswarm
        for i, subswarm_action in enumerate(self.subswarm_actions):
            subswarm_action(reformatted_action[i])

    def set_limits(self):
        self.config.lower_bound = np.array([
            subswarm.actual_low_limit_action_space
            for subswarm in self.subswarm_actions
        ], dtype=np.float32).flatten()

        self.config.upper_bound = np.array([
            subswarm.actual_high_limit_action_space
            for subswarm in self.subswarm_actions
        ], dtype=np.float32).flatten()


class ContinuousActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config

        self.action_names = ['Velocity Scaling Factor']
        # self.action_names = ['PBest Distance Threshold', 'Velocity Braking Factor']
        # self.action_names = ['Inertia', 'Social', 'Cognitive']
        self.practical_action_low_limit = [10]
        self.practical_action_high_limit = [190]
        # self.practical_action_low_limit = [0.75]
        # self.practical_action_high_limit = [1.25]

        #
        # self.actual_low_limit_action_space = [config.w_min, config.c_min, config.c_min]
        # self.actual_high_limit_action_space = [config.w_max, config.c_max, config.c_max]
        # self.practical_action_low_limit = [config.w_min, config.c_min, config.c_min]
        # self.practical_action_high_limit = [config.w_max, config.c_max, config.c_max]

        # self.practical_action_low_limit = [0, self.config.velocity_braking_min]
        # self.practical_action_high_limit = [self.config.distance_threshold_max, self.config.velocity_braking_max]
        #
        # self.actual_low_limit_action_space = [self.config.distance_threshold_min, self.config.velocity_braking_min]
        # self.actual_high_limit_action_space = [self.config.distance_threshold_max, self.config.velocity_braking_max]
        #
        self.actual_low_limit_action_space = [10]
        self.actual_high_limit_action_space = [190]
        # self.actual_low_limit_action_space = [0.75]
        # self.actual_high_limit_action_space = [1.25]
        #
        # self.actual_low_limit_action_space = [self.config.replacement_threshold_min]
        # self.actual_high_limit_action_space = [self.config.replacement_threshold_max]

    def __call__(self, action):
        actions = np.array(action)

        # self.swarm.w = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.c1 = np.clip(actions[1], self.practical_action_low_limit[1], self.practical_action_high_limit[1])
        # self.swarm.c2 = np.clip(actions[2], self.practical_action_low_limit[2], self.practical_action_high_limit[2])

        # self.swarm.pbest_replacement_threshold = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.distance_threshold = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.velocity_braking = np.clip(actions[1], self.practical_action_low_limit[1], self.practical_action_high_limit[1])
        # self.swarm.distance_threshold = np.clip(actions[0], 0, self.config.distance_threshold_max)
        self.swarm.abs_max_velocity = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.velocity_scaling_factor = np.clip(actions[1], self.practical_action_low_limit[1], self.practical_action_high_limit[1])
        #
        # if actions[0] > 0.50:
        #     self.reset_all_particles_keep_global_best()

    def set_limits(self):
        self.config.lower_bound = np.array(self.actual_low_limit_action_space, dtype=np.float32)
        self.config.upper_bound = np.array(self.actual_high_limit_action_space, dtype=np.float32)

    def reset_all_particles_keep_global_best(self):
        old_gbest_pos = self.swarm.P[np.argmin(self.swarm.P_vals)]
        old_gbest_val = np.min(self.swarm.P_vals)

        self.swarm.reinitialize()
        if type(self.swarm) == PSOSubSwarm:
            self.swarm.share_information_with_global_swarm = True

        # Keep Previous Solution before resetting.
        if old_gbest_val < self.swarm.gbest_val:
            self.swarm.gbest_pos = old_gbest_pos
            self.swarm.gbest_val = old_gbest_val