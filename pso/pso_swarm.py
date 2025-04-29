import numpy as np


class PSOSwarm:

    def __init__(self, objective_function, config):
        self.X = None  # Current Position of particles
        self.V = None  # Current Velocity of particles
        self.P = None  # Best Position of particles

        # PSO Parameters
        self.config = config
        self.w = config.w  # Inertia weight to prevent velocities becoming too large
        self.c1 = config.c1  # Social component Learning Factor
        self.c2 = config.c2  # Cognitive component Learning Factor
        self.function = objective_function
        self.dimension = config.dim
        self.swarm_size = config.swarm_size
        self.rangeF = config.rangeF
        self.perturb_velocities = False
        self.perturb_velocity_factor = None
        self.velocity_scaling_factor = 1.0
        self.perturb_velocity_particle_selection = None
        self.perturb_positions = False
        self.perturb_position_factor = None
        self.perturb_position_particle_selection = None

        # Threshold Params
        self.pbest_replacement_threshold = config.replacement_threshold
        self.distance_threshold = config.distance_threshold
        self.velocity_braking = config.velocity_braking

        # Observation Parameters
        self.num_swarm_obs_intervals = config.num_swarm_obs_intervals
        self.swarm_obs_interval_length = config.swarm_obs_interval_length
        self.iterations = self.num_swarm_obs_intervals * self.swarm_obs_interval_length

        # Set Constraints for clamping position and limiting velocity
        self.abs_max_velocity = self.rangeF
        self.abs_max_position = self.rangeF
        self.diagonal = 2.0 * self.rangeF * np.sqrt(self.dimension)

        self.velocity_magnitudes = None
        self.relative_fitnesses = None
        self.average_pbest_replacement_counts = None
        self.pbest_replacement_counts = None

        self.P_vals = None
        self.current_valuations = None
        self.gbest_val = float('inf')
        self.gbest_pos = None

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()

    def reinitialize(self):
        self._initialize()

    def _initialize(self):
        # Initialize 3 matrices for Current Position, Velocity, and Position of particles' best solution
        self.X = np.random.uniform(low=-1 * self.rangeF, high=self.rangeF, size=(self.swarm_size, self.dimension))
        self.V = np.full((self.swarm_size, self.dimension), 0.0)
        self.P = self.X
        self.P_vals = None

        # Reset the adjustable parameters to starting values
        self.w = self.config.w
        self.c1 = self.config.c1
        self.c2 = self.config.c2
        self.pbest_replacement_threshold = self.config.replacement_threshold
        self.distance_threshold = self.config.distance_threshold
        self.velocity_scaling_factor = 1.0
        self.velocity_braking = self.config.velocity_braking
        self.abs_max_velocity = self.rangeF

        self.initialize_stored_counts()

        # Record the initialized particle and global best solutions
        self.update_swarm_valuations_and_bests()

    def initialize_stored_counts(self):
        self.velocity_magnitudes = None
        self.relative_fitnesses = None

        self.current_valuations = None
        self.gbest_val = float('inf')
        self.gbest_pos = None

        # Static class variables to track P_vals replacements
        self.pbest_replacement_counts = np.zeros(self.swarm_size)
        self.pbest_replacement_batchcounts = np.zeros((self.num_swarm_obs_intervals, self.swarm_size))
        self.average_pbest_replacement_counts = np.zeros(self.swarm_size)

    def update_swarm_valuations_and_bests(self):
        self.P_vals = self.eval(self.P)  # Vector of particle's current valuation of its best position based on Function Eval
        self.current_valuations = self.eval(self.P)  # Vector of particle's current value of its position based on Function Eval
        self.update_gbest()

        self.update_relative_fitnesses()
        self.update_velocity_maginitude()

    # def _inject_random_perturbations_to_velocities(self, selection_type, factor):
    #     self.update_velocity_maginitude()
    #     avg_velocity = np.mean(self.velocity_magnitudes)
    #
    #     # Determine the particles to select based on the selection type
    #     if selection_type == 0:  # Slow half particles
    #         selected_particles = self.velocity_magnitudes < avg_velocity
    #     elif selection_type == 1:  # Fast half particles
    #         selected_particles = self.velocity_magnitudes >= avg_velocity
    #     elif selection_type == 2:  # All particles
    #         selected_particles = np.ones(self.velocity_magnitudes.shape, dtype=bool)  # Select all particles
    #     else:
    #         raise ValueError("Invalid selection type")
    #
    #     selected_particles_reshaped = selected_particles[:, np.newaxis]  # Reshape for broadcasting
    #
    #     # Generate random perturbations based on the factor
    #     random_perturbations = np.random.uniform(
    #         low=-factor * self.abs_max_velocity,
    #         high=factor * self.abs_max_velocity,
    #         size=self.V.shape
    #     )
    #
    #     # Apply perturbations to the selected particles
    #     perturbed_velocities = self.V + random_perturbations
    #     perturbed_velocities = np.clip(perturbed_velocities, -self.abs_max_velocity, self.abs_max_velocity)
    #
    #     # Update velocities and the swarm
    #     self.V = np.where(selected_particles_reshaped, perturbed_velocities, self.V)
    #     self.update_swarm_valuations_and_bests()

    def update_relative_fitnesses(self):
        self.relative_fitnesses = (self.P_vals - self.gbest_val) / np.abs(self.P_vals)

    def update_velocity_maginitude(self):
        self.velocity_magnitudes = np.linalg.norm(self.V, axis=1)

    def update_average_pbest_replacement_counts(self):
        # Calculate the average batch count for each particle
        self.average_pbest_replacement_counts = np.mean(self.pbest_replacement_batchcounts, axis=0)

    def get_observation(self):
        self.update_relative_fitnesses()
        self.update_velocity_maginitude()
        self.update_average_pbest_replacement_counts()

        velocity_magnitudes_norm = self.velocity_magnitudes / self.rangeF
        relative_fitnesses_norm = np.tanh(self.relative_fitnesses)
        pbest_counts_norm = self.average_pbest_replacement_counts / self.swarm_obs_interval_length


        obs_stack = np.column_stack([
            velocity_magnitudes_norm,
            relative_fitnesses_norm,
            pbest_counts_norm
        ])

        # Flatten to ensure the observations are in the required 1D format
        obs = obs_stack.flatten()

        # Add in the current replacement threshold
        # obs = np.append(obs, self.pbest_replacement_threshold)
        # obs = np.append(obs, self.distance_threshold)
        # obs = np.append(obs, self.velocity_braking)distance_threshold

        # # # Compute diversity as average distance to centroid
        swarm_centroid = np.mean(self.X, axis=0)
        diversity = np.mean(np.linalg.norm(self.X - swarm_centroid, axis=1))
        obs = np.append(obs, diversity)

        return obs

    def get_swarm_observation(self):
        return {
            "w": self.w,
            "c1": self.c1,
            "c2": self.c2,
            "abs_max_velocity": self.abs_max_velocity,
            "abs_max_position": self.abs_max_position,
            "gbest_val": self.gbest_val,
        }

    def get_current_best_fitness(self):
        return self.gbest_val
    
    def get_current_best_fitnesses(self):
        return self.P_vals

    def eval(self, X):
        return self.function.Y_matrix(np.array(X).astype(float))

    def update_velocities(self, leader):
        # Generate social and cognitive components of velocity matrix update using np.random
        social = self.c1 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (leader - self.X)
        cognitive = self.c2 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (self.P - self.X)

        # Update new velocity with old velocity*inertia plus component matrices
        self.V = self.w * self.V + social + cognitive

        self.V = self.V * self.velocity_scaling_factor

        # self.V = np.clip(self.V, -self.abs_max_velocity, self.abs_max_velocity)
        max_component_per_particle = np.max(np.abs(self.V), axis=1)  # shape: (swarm_size,)
        exceed_mask = max_component_per_particle > self.abs_max_velocity # Find particles that exceed the max allowed

        # Only scale those particles
        scaling_factors = self.abs_max_velocity / max_component_per_particle[exceed_mask]  # shape: (num_exceed,)
        self.V[exceed_mask] = self.V[exceed_mask] * scaling_factors[:, np.newaxis]

        # # Apply breaking factor to the velocity

    def update_positions(self):
        # Clamp position inside boundary and reflect them in case they are out of the boundary based on:
        # S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization," IEEE TEC: 17(2), 2013, pp. 259-271

        self.X = self.X + self.V # Add velocity to current position

        # Identify positions outside the boundaries, and reflect the out-of-bounds positions
        out_of_bounds = np.logical_or(self.X < -self.abs_max_position, self.X > self.abs_max_position)
        self.X = np.where(out_of_bounds, np.sign(self.X) * 2 * self.abs_max_position - self.X, self.X)
        self.V = np.where(out_of_bounds, 0, self.V)  # Out-of-bounds velocities are set to 0

        # Use function evaluation for each particle (vector) in Swarm to provide value for each position in X.
        self.current_valuations = self.eval(self.X)

    def update_pbests(self):
        improved_particles = self.current_valuations < self.P_vals  # Update each Particle's best position for each particle index
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.P_vals = np.where(improved_particles, self.current_valuations, self.P_vals)

        self.pbest_replacement_counts += improved_particles  # Update P_vals replacement counter

    def update_pbest_with_non_elitist_selection(self):
        improved_particles = self.pbest_replacement_threshold * self.current_valuations < self.P_vals
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.P_vals = np.where(improved_particles, self.current_valuations, self.P_vals)

        self.pbest_replacement_counts += improved_particles  # Update pbest_val replacement counts
    #
    def update_pbest_with_distance_threshold(self):
        # Compute Euclidean distances between the current position and the particle's best
        distances = np.linalg.norm(self.X - self.P, axis=1)

        # Determine for each particle if the new candidate is better than its current p_best, AND if the new candidate is not a very small (local) improvement
        # (i.e. distance is above the replacement threshold)
        threshold = self.distance_threshold * self.diagonal
        improved_particles = (self.current_valuations < self.P_vals) & (distances > threshold)

        # improved_particles = self.pbest_replacement_threshold * self.current_valuations < self.P_vals
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.P_vals = np.where(improved_particles, self.current_valuations, self.P_vals)

        self.pbest_replacement_counts += improved_particles  # Update pbest_val replacement counter

    def decay_parameters(self, obs_interval, iteration_idx):
        # Decay the replacement threshold over time back to 1
        total_iterations = self.num_swarm_obs_intervals * self.swarm_obs_interval_length
        current_step = (obs_interval * self.swarm_obs_interval_length) + iteration_idx
        linear_decay_rate = 2 * 1 / (total_iterations - current_step)

        # self.pbest_replacement_threshold += (1 - self.pbest_replacement_threshold) * linear_decay_rate
        # self.pbest_replacement_threshold = min(1, self.pbest_replacement_threshold)

        # self.distance_threshold += (0 - self.distance_threshold) * linear_decay_rate
        #
        # self.velocity_braking += (1 - self.velocity_braking) * linear_decay_rate

        self.velocity_scaling_factor += (1 - self.velocity_scaling_factor) * linear_decay_rate
        x=1

    def update_gbest(self):
        self.gbest_pos = self.P[np.argmin(self.P_vals)]  # Vector of globally best visited position.
        self.gbest_val = np.min(self.P_vals)  # Single valuation of the global best position

    def store_and_reset_batch_counts(self, obs_interval_idx):
        self.pbest_replacement_batchcounts[obs_interval_idx] = self.pbest_replacement_counts
        self.pbest_replacement_counts = np.zeros(self.swarm_size)

    def optimize(self):
        for obs_interval_idx in range(self.num_swarm_obs_intervals):
            for iteration_idx in range(self.swarm_obs_interval_length):
                self.optimize_single_iteration(self.gbest_pos, obs_interval_idx, iteration_idx)
            self.store_and_reset_batch_counts(obs_interval_idx)

    def optimize_single_iteration(self, global_leader, obs_interval, iteration_idx):
        # if obs_interval == 0 and iteration_idx < self.swarm_obs_interval_length * 0.10:
        if obs_interval == 0 and iteration_idx == 0:  # Only perturb velocities in the first iteration of the first observation interval
            self.perturb_velocities = False
            self.perturb_positions = False

        self.update_velocities(global_leader)  # Input global leader particle position

        # if self.perturb_velocities:
        #     self._inject_random_perturbations_to_velocities(self.perturb_velocity_particle_selection, self.perturb_velocity_factor)

        self.update_positions()
        self.update_pbests()
        # self.update_pbest_with_non_elitist_selection()
        # self.update_pbest_with_distance_threshold()
        self.update_gbest()

        self.decay_parameters(obs_interval, iteration_idx)







