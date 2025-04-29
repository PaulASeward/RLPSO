import numpy as np
from enum import Enum
from pso.pso_multiswarm import PSOSubSwarm


class DiscreteMultiswarmActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config
        self.subswarm_actions = [DiscreteActions(sub_swarm, config) for sub_swarm in self.swarm.sub_swarms]
        self.action_names = ['Do Nothing'] + [
            f"SubSwarm {i + 1} {action_name}"
            for i, subswarm in enumerate(self.subswarm_actions)
            for action_name in subswarm.action_names
        ]

        self.config.num_actions = len(self.action_names)

    def __call__(self, action):
        # Do nothing
        if action == 0:
            return
        action -= 1

        # Dividend operation to get the action for the respective subswarm
        subswarm_index = action // len(self.subswarm_actions[0].action_names)
        action_index = action % len(self.subswarm_actions[0].action_names)

        # Call the respective subswarm action
        self.subswarm_actions[subswarm_index](action_index)



class DiscreteActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config

        # self.action_methods = {
        #     0: self.reset_all_particles_keep_global_best,
        # }

        # self.action_names = ['Do nothing', 'Increase inertia', 'Decrease inertia', 'Increase social factor', 'Decrease social factor', 'Reset Slow Particles', 'Reset All Particles Keep Global Best']
        # self.action_methods = {
        #     0: self.do_nothing,
        #     1: self.increase_inertia,
        #     2: self.decrease_inertia,
        #     3: self.increase_social_factor,
        #     4: self.decrease_social_factor,
        #     5: self.reset_slow_particles,
        #     6: self.reset_all_particles_keep_global_best,
        # }

        # self.action_names = ['Do nothing', 'Increase all velocities', 'Decrease all velocities', 'Increase max velocity', 'Decrease max velocity', 'Speed up slower half', 'Slow down faster half',  'Reset Slower half', 'Reset All Particles Keep Global Best']
        # self.action_methods = {
        #     0: self.do_nothing,
        #     1: self.increase_all_velocities,
        #     2: self.decrease_all_velocities,
        #     3: self.increase_max_velocity,
        #     4: self.decrease_max_velocity,
        #     5: self.increase_velocities_of_slow_velocities,
        #     6: self.decrease_velocities_of_fast_particles,
        #     7: self.reset_slow_particles,
        #     8: self.reset_all_particles_keep_global_best,
        # }

        # self.action_names = [
        #     'Reset slow particles velocity to random keep position',
        #     'Reset fast particles velocity to random keep position',
        #     'Reset slow particles velocity to random reset position',
        #     'Reset slow velocity to zero reset position'
        # ]
        # self.action_methods = {
        #     0: self.reset_slow_particles_velocity_to_random_keep_position,
        #     1: self.reset_fast_particles_velocity_to_random_keep_position,
        #     2: self.reset_slow_particles_velocity_to_random_reset_position,
        #     3: self.reset_slow_velocity_to_zero_reset_position,
        # }

        # self.action_names = ['Reset slow particles velocity to random keep position',
        #                      'Reset fast particles velocity to random keep position',
        #                      'Reset all particles velocity to random keep position',
        #                      'Reset slow particles velocity to random reset position',
        #                      'Reset fast particles velocity to random reset position',
        #                      'Reset all particles velocity to random reset position',
        #                      'Reset slow particles velocity to zero reset position',
        #                      'Reset fast particles velocity to zero reset position',
        #                      'Reset all particles velocity to zero reset position']
        # self.action_methods = {
        #     0: self.reset_slow_particles_velocity_to_random_keep_position,
        #     1: self.reset_all_particles_velocity_to_random_keep_position,
        #     2: self.reset_fast_particles_velocity_to_random_keep_position,
        #     3: self.reset_slow_particles_velocity_to_random_reset_position,
        #     4: self.reset_particles_velocity_to_random_reset_position,
        #     5: self.reset_fast_particles_velocity_to_random_reset_position,
        #     6: self.reset_slow_velocity_to_zero_reset_position,
        #     7: self.reset_particles_velocity_to_zero_reset_position,
        #     8: self.reset_fast_velocity_to_zero_reset_position,
        # }

        self.action_methods = {
            0: self.reset_slow_particles,
            1: self.reset_all_particles_keep_global_best,
            2: self.reset_all_particles_without_memory_sharing,
            3: self.reshare_information_with_global_swarm
        }

        self.action_names = [ 'Reset slow particles',
                              'Reset all particles with preserved information',
                              'Reset all particles without memory',
                              'Reshare information with global swarm'
                             ]

    def __call__(self, action):
        if not isinstance(action, int):
            action = action.item()
        action_method = self.action_methods.get(action, lambda: None)
        action_method()

    def do_nothing(self):
        return

    def reshare_information_with_global_swarm(self):
        if type(self.swarm) == PSOSubSwarm:
            self.swarm.share_information_with_global_swarm = True

        self.swarm.update_superswarm_gbest()

    def reset_all_particles_without_memory_sharing(self):
        self.swarm.reinitialize()

        if type(self.swarm) == PSOSubSwarm:
            self.swarm.share_information_with_global_swarm = False

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

    def reset_all_particles_keep_personal_best(self):
        old_pbest_pos = self.swarm.P

        self.swarm.reinitialize()

        self.swarm.P = old_pbest_pos
        self.swarm.update_swarm_valuations_and_bests()

    def reset_slow_particles(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF,
                                                  size=(self.swarm.swarm_size, self.swarm.dimension))
        replacement_velocities = np.full((self.swarm.swarm_size, self.swarm.dimension), 0)
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        self.swarm.X = np.where(slow_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(slow_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = np.where(slow_particles_reshaped, self.swarm.X, self.swarm.P)

        self.swarm.update_swarm_valuations_and_bests()

    def _reset_particles_position_using_lattice(self):
        grid_points = int(np.cbrt(self.swarm.swarm_size))  # Assuming cubic grid
        lattice = np.linspace(-1 * self.swarm.rangeF, self.swarm.rangeF, grid_points)
        positions = np.array(np.meshgrid(*([lattice] * self.swarm.dimension))).T.reshape(-1, self.swarm.dimension)
        np.random.shuffle(positions)
        self.swarm.X[:positions.shape[0]] = positions[:self.swarm.swarm_size]

    def _reset_particles_position_to_random(self):
        self.swarm.X = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))

    def _reset_particles_velocity_to_random(self):
        self.swarm.V = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))

    def _reset_particles_velocity_to_zero(self):
        self.swarm.V = np.full((self.swarm.swarm_size, self.swarm.dimension), 0.0)

    def _forget_memory(self):
        self.swarm.P = self.swarm.X
        self.swarm.P_vals = None

    def reset_all_particles_velocity_to_random_keep_position(self):
        self._reset_particles_velocity_to_random()

    def reset_slow_particles_velocity_to_random_keep_position(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        replacement_velocities = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X
        self.swarm.V = np.where(slow_particles_reshaped, replacement_velocities, self.swarm.V)

    def reset_fast_particles_velocity_to_random_keep_position(self):
        avg_velocity = self._calculate_average_velocity()
        fast_particles = self.swarm.velocity_magnitudes > avg_velocity
        replacement_velocities = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))
        fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match self.swarm.X
        self.swarm.V = np.where(fast_particles_reshaped, replacement_velocities, self.swarm.V)

    def reset_particles_velocity_to_random_reset_position(self):
        self._reset_particles_position_to_random()
        self._reset_particles_velocity_to_random()
        self.swarm.update_swarm_valuations_and_bests()

    def reset_slow_particles_velocity_to_random_reset_position(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X
        replacement_velocities = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))

        self.swarm.X = np.where(slow_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(slow_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = np.where(slow_particles_reshaped, self.swarm.X, self.swarm.P)

        self.swarm.update_swarm_valuations_and_bests()

    def reset_fast_particles_velocity_to_random_reset_position(self):
        avg_velocity = self._calculate_average_velocity()
        fast_particles = self.swarm.velocity_magnitudes > avg_velocity
        fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        replacement_velocities = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))

        self.swarm.X = np.where(fast_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(fast_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = np.where(fast_particles_reshaped, self.swarm.X, self.swarm.P)

        self.swarm.update_swarm_valuations_and_bests()

    def reset_particles_velocity_to_zero_reset_position(self):
        self._reset_particles_position_to_random()
        self._reset_particles_velocity_to_zero()
        self.swarm.update_swarm_valuations_and_bests()

    def reset_slow_velocity_to_zero_reset_position(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        slow_particles_reshaped = slow_particles[:, np.newaxis]

        replacement_velocities = np.full((self.swarm.swarm_size, self.swarm.dimension), 0.0)
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))

        self.swarm.X = np.where(slow_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(slow_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = np.where(slow_particles_reshaped, self.swarm.X, self.swarm.P)
        self.swarm.update_swarm_valuations_and_bests()

    def reset_fast_velocity_to_zero_reset_position(self):
        avg_velocity = self._calculate_average_velocity()
        fast_particles = self.swarm.velocity_magnitudes > avg_velocity
        fast_particles_reshaped = fast_particles[:, np.newaxis]

        replacement_velocities = np.full((self.swarm.swarm_size, self.swarm.dimension), 0.0)
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))

        self.swarm.X = np.where(fast_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(fast_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = np.where(fast_particles_reshaped, self.swarm.X, self.swarm.P)

        self.swarm.update_swarm_valuations_and_bests()

    def increase_all_velocities(self):
        self.swarm.velocity_scaling_factor *= 1.10

    def decrease_all_velocities(self):
        self.swarm.velocity_scaling_factor *= 0.90

    def _calculate_average_velocity(self):
        self.swarm.update_velocity_maginitude()
        return np.mean(self.swarm.velocity_magnitudes)

    def inject_random_perturbations_to_velocities(self, selection_type, factor):
        self.swarm.perturb_velocities = True
        self.swarm.perturb_velocity_factor = factor
        self.swarm.perturb_velocity_particle_selection = selection_type

    def inject_small_perturbations_to_slow_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=0, factor=0.05)

    def inject_medium_perturbations_to_slow_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=0, factor=0.20)

    def inject_large_perturbations_to_slow_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=0, factor=0.50)

    def inject_small_perturbations_to_fast_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=1, factor=0.05)

    def inject_medium_perturbations_to_fast_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=1, factor=0.20)

    def inject_large_perturbations_to_fast_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=1, factor=0.50)

    def inject_small_perturbations_to_all_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=2, factor=0.05)

    def inject_medium_perturbations_to_all_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=2, factor=0.20)

    def inject_large_perturbations_to_all_particles(self):
        self.inject_random_perturbations_to_velocities(selection_type=2, factor=0.50)

    def increase_velocities_of_slow_velocities(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        faster_velocities = self.swarm.V * 1.10
        faster_velocities = np.clip(faster_velocities, -self.swarm.abs_max_velocity, self.swarm.abs_max_velocity)
        self.swarm.V = np.where(slow_particles_reshaped, faster_velocities, self.swarm.V)

    def decrease_velocities_of_fast_particles(self):
        avg_velocity = self._calculate_average_velocity()
        fast_particles = self.swarm.velocity_magnitudes > avg_velocity
        fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        slower_velocities = self.swarm.V * 0.90
        self.swarm.V = np.where(fast_particles_reshaped, slower_velocities, self.swarm.V)

    def increase_max_velocity(self):
        self.swarm.abs_max_velocity *= 1.10
        self.swarm.abs_max_velocity = np.clip(self.swarm.abs_max_velocity, self.config.v_min, self.config.v_max)

    def decrease_max_velocity(self):
        self.swarm.abs_max_velocity *= 0.90
        self.swarm.abs_max_velocity = np.clip(self.swarm.abs_max_velocity, self.config.v_min, self.config.v_max)

    def increase_social_factor(self):
        self.swarm.c1 *= 1.10  # Social component
        self.swarm.c1 = np.clip(self.swarm.c1, self.config.c_min, self.config.c_max)

        self.swarm.c2 *= 0.90  # Cognitive component
        self.swarm.c2 = np.clip(self.swarm.c2, self.config.c_min, self.config.c_max)

    def increase_inertia(self):
        self.swarm.w *= 1.10
        self.swarm.w = np.clip(self.swarm.w, self.config.w_min, self.config.w_max)

    def decrease_inertia(self):
        self.swarm.w *= 0.90
        self.swarm.w = np.clip(self.swarm.w, self.config.w_min, self.config.w_max)

    def decrease_social_factor(self):
        self.swarm.c1 *= 0.90
        self.swarm.c1 = np.clip(self.swarm.c1, self.config.c_min, self.config.c_max)

        self.swarm.c2 *= 1.10
        self.swarm.c2 = np.clip(self.swarm.c2, self.config.c_min, self.config.c_max)

    # Threshold actions to promote exploration vs exploitation
    def decrease_pbest_replacement_threshold(self):
        self.swarm.pbest_replacement_threshold *= self.swarm.pbest_replacement_threshold_decay
        self.swarm.pbest_replacement_threshold = np.clip(self.swarm.pbest_replacement_threshold,
                                                         self.config.pbest_replacement_threshold_min,
                                                         self.config.pbest_replacement_threshold_max)

    def increase_pbest_replacement_threshold(self):
        self.swarm.pbest_replacement_threshold *= 1.10
        self.swarm.pbest_replacement_threshold = np.clip(self.swarm.pbest_replacement_threshold,
                                                         self.config.pbest_replacement_threshold_min,
                                                         self.config.pbest_replacement_threshold_max)