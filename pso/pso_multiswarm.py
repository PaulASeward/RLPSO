import numpy as np
from pso.pso_swarm import PSOSwarm

class Particle:
    def __init__(self, x, v, p, P_val, current_valuation, velocity_magnitude, relative_fitness, average_pbest_replacement_counts):
        self.x = x
        self.v = v
        self.p = p
        self.P_val = P_val
        self.velocity_magnitude = velocity_magnitude
        self.current_valuation = current_valuation
        self.relative_fitness = relative_fitness
        self.average_pbest_replacement_counts = average_pbest_replacement_counts


class ParticleDataMetaData:
    def __init__(self, particle, current_sub_swarm_idx, current_idx_within_sub_swarm, target_sub_swarm_idx):
        self.particle = particle
        self.sub_swarm = current_sub_swarm_idx
        self.idx = current_idx_within_sub_swarm
        self.target_sub_swarm = target_sub_swarm_idx


class PSOSubSwarm(PSOSwarm):
    def __init__(self, objective_function, config, gbest_val, gbest_pos):
        self.gbest_val = gbest_val
        self.gbest_pos = gbest_pos

        self.share_information_with_global_swarm = True
        self.sub_swarm_gbest_pos = None
        self.sub_swarm_gbest_val = float('inf')
        super().__init__(objective_function, config)


    def update_gbest(self):
        self.sub_swarm_gbest_val = np.min(self.P_vals)
        self.sub_swarm_gbest_pos = self.P[np.argmin(self.P_vals)]
        if self.share_information_with_global_swarm:
            self.update_superswarm_gbest()

    def update_superswarm_gbest(self):
        if self.sub_swarm_gbest_val < self.gbest_val:
            self.gbest_val = self.sub_swarm_gbest_val
            self.gbest_pos = self.sub_swarm_gbest_pos

    def get_current_best_fitness(self):
        return self.sub_swarm_gbest_val

    def get_particle(self, particle_index: int):
        return Particle(self.X[particle_index],
                        self.V[particle_index],
                        self.P[particle_index],
                        self.P_vals[particle_index],
                        self.current_valuations[particle_index],
                        self.velocity_magnitudes[particle_index],
                        self.relative_fitnesses[particle_index],
                        self.average_pbest_replacement_counts[particle_index])

    def add_particle(self, particle: Particle, particle_index: int):
        self.X[particle_index] = particle.x
        self.V[particle_index] = particle.v
        self.P[particle_index] = particle.p
        self.P_vals[particle_index] = particle.P_val
        self.current_valuations[particle_index] = particle.current_valuation
        self.velocity_magnitudes[particle_index] = particle.velocity_magnitude
        self.relative_fitnesses[particle_index] = particle.relative_fitness
        self.average_pbest_replacement_counts[particle_index] = particle.average_pbest_replacement_counts

    def swap_in_particles(self, incoming_particles: [ParticleDataMetaData], outgoing_particles: [ParticleDataMetaData]):
        assert len(incoming_particles) == len(outgoing_particles), "Incoming and Outgoing Particle Counts must be equal"

        for i, incoming_particle_data in enumerate(incoming_particles):
            self.add_particle(incoming_particle_data.particle, outgoing_particles[i].idx)


class PSOMultiSwarm:

    def __init__(self, objective_function, config):
        # PSO Parameters
        self.config = config
        self.swarm_size = config.swarm_size
        self.objective_function = objective_function
        # clone config to avoid changing the original config
        self.sub_swarm_config = config.clone()
        self.sub_swarm_size = config.swarm_size // config.num_sub_swarms
        self.sub_swarm_config.swarm_size = self.sub_swarm_size
        self.sub_swarm_config.is_sub_swarm = True
        self.dim = config.dim
        self.num_sub_swarms = config.num_sub_swarms

        self.gbest_val = float('inf')
        self.gbest_pos = None

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()

    def reinitialize(self):
        for sub_swarm in self.sub_swarms:
            sub_swarm.reinitialize()

        self.update_swarm_valuations_and_bests()

    def _initialize(self):
        self.sub_swarms = [PSOSubSwarm(self.objective_function, self.sub_swarm_config, gbest_val=self.gbest_val, gbest_pos=self.gbest_pos) for _ in range(self.num_sub_swarms)]

    def update_swarm_valuations_and_bests(self):
        for sub_swarm in self.sub_swarms:
            sub_swarm.update_swarm_valuations_and_bests()

        self.update_gbest()

    def get_observation(self):
        sub_swarm_observations = [sub_swarm.get_observation() for sub_swarm in self.sub_swarms]
        multiswarm_observation = np.concatenate(sub_swarm_observations, axis=0)
        return multiswarm_observation

    def get_swarm_observation(self):
        swarm_observation = {}
        for i, sub_swarm in enumerate(self.sub_swarms):
            swarm_observation[f"sub_swarm_{i}"] = sub_swarm.get_observation()
        return swarm_observation

    def get_current_best_fitness(self):
        return self.gbest_val

    def update_gbest(self):
        sub_swarms_with_sharing_info = [sub_swarm for sub_swarm in self.sub_swarms if sub_swarm.share_information_with_global_swarm]
        if len(sub_swarms_with_sharing_info) == 0:
            return

        best_subswarm_fitnesses = [sub_swarm.get_current_best_fitness() for sub_swarm in sub_swarms_with_sharing_info]
        best_sub_swarm_idx = np.argmin(best_subswarm_fitnesses)
        self.gbest_val = sub_swarms_with_sharing_info[best_sub_swarm_idx].gbest_val
        self.gbest_pos = sub_swarms_with_sharing_info[best_sub_swarm_idx].gbest_pos

    def reorganize_swarms_by_rank(self):
        # Identify only those subswarms that share information with the global swarm
        available_subswarms = [s for s in self.sub_swarms if s.share_information_with_global_swarm]
        if not available_subswarms:
            return

        all_particles = []
        for sub_swarm in available_subswarms:
            for idx in range(self.sub_swarm_size):
                all_particles.append(sub_swarm.get_particle(idx))
        all_particles.sort(key=lambda p: p.P_val)

        # Distribute them back: the best sub_swarm_size go to subswarm 0, the next group go to subswarm 1, etc.
        start_idx = 0
        for sub_swarm in available_subswarms:
            for local_idx in range(self.sub_swarm_size):
                sub_swarm.add_particle(all_particles[start_idx + local_idx], local_idx)
            start_idx += self.sub_swarm_size
            sub_swarm.update_gbest()

        x=1

    def reorganize_swarms_by_swaps(self):
        # Reorder the sub_swarms position based on average fitness
        self.sub_swarms = sorted(self.sub_swarms, key=lambda sub_swarm: sub_swarm.sub_swarm_gbest_val)

        # Reorganize particles into subswarms, grouped by relative fitness.
        # Call each sub_swarm to return a value of their p_best fitness.
        # Make a one-dimensional array of all p_best values
        # Generate a list representing the rank of each particles pbest value in the global swarm
        P_vals_flattened = np.array([sub_swarm.P_vals for sub_swarm in self.sub_swarms]).flatten()
        global_ranks = np.argsort(P_vals_flattened)
        global_swarm_rank_indices = np.empty_like(global_ranks)  # Initialize array for ranks
        global_swarm_rank_indices[global_ranks] = np.arange(len(P_vals_flattened))

        IN = "incoming"
        OUT = "outgoing"

        incoming_and_outgoing_particles = [{IN: [], OUT: []} for _ in range(self.num_sub_swarms)]

        for global_index, target_sub_swarm_idx in enumerate(global_swarm_rank_indices // self.sub_swarm_size):
            current_sub_swarm_idx = global_index // self.sub_swarm_size
            current_idx_within_sub_swarm = global_index % self.sub_swarm_size

            if current_sub_swarm_idx != target_sub_swarm_idx: # If the particle is not in the correct sub_swarm
                particle = self.sub_swarms[current_sub_swarm_idx].get_particle(current_idx_within_sub_swarm)
                particle_meta_data = ParticleDataMetaData(particle, current_sub_swarm_idx, current_idx_within_sub_swarm, target_sub_swarm_idx)

                incoming_and_outgoing_particles[current_sub_swarm_idx][OUT].append(particle_meta_data)
                incoming_and_outgoing_particles[target_sub_swarm_idx][IN].append(particle_meta_data)

        # Swap incoming particles with outgoing particles for each sub_swarm
        for sub_swarm_idx, incoming_outgoing_dict in enumerate(incoming_and_outgoing_particles):
            self.sub_swarms[sub_swarm_idx].swap_in_particles(incoming_outgoing_dict[IN], incoming_outgoing_dict[OUT])

    def optimize(self):
        for obs_interval_idx in range(self.config.num_swarm_obs_intervals):
            for iteration_idx in range(self.config.swarm_obs_interval_length):
                for sub_swarm in self.sub_swarms:
                    if sub_swarm.share_information_with_global_swarm:
                        leader = self.gbest_pos
                        if leader is None:
                            x=1
                        # TODO: Debug how the self.gbest_pos is None
                    else:
                        leader = sub_swarm.sub_swarm_gbest_pos

                        if leader is None:
                            x=1
                    sub_swarm.optimize_single_iteration(leader, obs_interval_idx, iteration_idx)

                self.update_gbest()
            for sub_swarm in self.sub_swarms:
                sub_swarm.store_and_reset_batch_counts(obs_interval_idx)

        self.update_gbest()
        # self.reorganize_swarms_by_swaps()
        self.reorganize_swarms_by_rank()








