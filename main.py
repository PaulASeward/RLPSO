import os

from agents.dqn_agent import DQNAgent
from agents.drqn_agent import DRQNAgent
from agents.ddpg_agent import DDPGAgent
from agents.ddrpg_agent import DDRPGAgent
from agents.ppo_agent import PPOAgent
from config import Config
import argparse


class Main:
    def __init__(self, config):
        agent_mapping = {
            "DQN": DQNAgent,
            "DRQN": DRQNAgent,
            "DDPG": DDPGAgent,
            "DDRPG": DDRPGAgent,
            "PPO": PPOAgent
        }

        self.agent = agent_mapping.get(config.network_type, DRQNAgent)(config)

    def train(self):
        if config.load_checkpoint_dir is not None:
            print("Loading models and buffer from checkpoint: ", config.load_checkpoint_dir)
            step_to_resume = int(config.load_checkpoint.split("_")[-1])
            self.agent.load_from_checkpoint(step_to_resume)

        self.agent.get_actions()
        self.agent.train()
        self.plot()

    def plot(self):
        self.agent.results_logger.plot_results()

    def test(self):
        if config.load_checkpoint_dir is not None:
            print("Loading models from checkpoint: ", config.load_checkpoint_dir)
            self.agent.load_models()
        else:
            if not config.train:
                raise ValueError("An agent must either be trained or loaded to test. Please specify a checkpoint to load, or train an agent first.")

        self.agent.test(step=self.agent.config.train_steps, number_of_tests=self.agent.config.num_final_tests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN Agent on PSO Algorithm")
    parser.add_argument("--network_type", type=str, default="DRQN", help="Type of the network to build, can either be 'DQN', 'DDPG',  or 'DRQN'")
    parser.add_argument("--swarm_algorithm", type=str, default="PMSO", help="The metaheuristic swarm algorithm to use. Currently only PSO or PMSO is supported")
    parser.add_argument("--func_num", type=int, default=14, help="The function number to optimize. Good functions to evaluate are 6,10,11,14,19")
    parser.add_argument("--num_subswarms", type=int, default=2, help="The number of sub swarms. Algorithm must be PMSO Default is 5.")
    parser.add_argument("--num_actions", type=int, default=9, help="The number of actions to choose from in the action space. Default is 5.")
    parser.add_argument("--action_dimensions", type=int, default=4, help="The number of actions to choose from in the action space. Each subswarm will have this many dimensions.")
    parser.add_argument("--train", type=bool, default=False, help="Whether to train an agent or to replot an agent's results")
    parser.add_argument("--test", type=bool, default=False, help="Whether to evaluate an agent from a trained/loaded state")
    parser.add_argument("--num_final_tests", type=int, default=100, help="How many times to evaluate an pretrained agent")
    parser.add_argument("--mock", type=bool, default=False, help="To use a mock data environment for evaluating")
    parser.add_argument("--priority_replay", type=bool, default=False, help="To use a priority replay buffer for training")
    parser.add_argument("--over_sample_exploration", type=int, default=3, help="How much, if any, to oversample initial exploration to lean learning towards intitial examples.  None, 0, keep same behavior. 1+ will oversample exploration")
    parser.add_argument("--steps", type=int, default=2000, help="number of iterations to train")
    parser.add_argument("--save_models", type=bool, default=True, help="Whether to save the models")
    parser.add_argument("--save_buffer", type=bool, default=True, help="Whether to save the experience buffer")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint to previously saved models. (e.g., 'step_100') Algorithms with two models will load both ")
    args, remaining = parser.parse_known_args()

    config = Config()
    config.train = args.train
    config.test = args.test
    config.num_final_tests = args.num_final_tests
    config.use_mock_data = args.mock
    config.use_priority_replay = args.priority_replay
    config.save_models = args.save_models
    config.load_checkpoint = args.load_checkpoint
    config.save_buffer = args.save_buffer

    assert args.swarm_algorithm in ["PSO", "PMSO"], "Please specify a swarm_algorithm of either PSO or PMSO"
    assert args.network_type in ["DQN", "DRQN", "DDPG", "DDRPG", "PPO"], "Please specify a network_type of either DQN, DRQN, or DDPG"
    assert args.func_num in list(range(1, 29)), "Please specify a func_num from 1-28"
    if args.swarm_algorithm == "PMSO":
        assert args.num_subswarms in [1,2,5,10,25,50], "Please specify a num_subswarms from 1,2,5,10,25,50"

    config.update_properties(network_type=args.network_type, swarm_algorithm=args.swarm_algorithm, func_num=args.func_num, num_actions=args.num_actions, load_checkpoint=args.load_checkpoint,
                             action_dimensions=args.action_dimensions, num_subswarms=args.num_subswarms, swarm_size=50, dimensions=30, num_episodes=20,
                             num_swarm_obs_intervals=10, swarm_obs_interval_length=30,
                             train_steps=args.steps, over_sample_exploration=args.over_sample_exploration)
    assert config.dim in [2, 5, 10, 20, 20, 30, 40, 50, 60, 70, 80, 90, 100], "Please specify a dim from 2,5,10,20,30,40,50,60,70,80,90,100"

    print("==== Experiment: ", config.experiment)
    print("==== Args used:")
    print(args)
    print("==== Remaining (Unknown) Args:")
    print(remaining)

    main = Main(config)

    if config.train:
        print(">> Training mode. Number of Steps to Train:", config.train_steps)
        print(">> Evaluation Interval:", config.eval_interval)
        print(">> Logging Interval:", config.log_interval)
        print(">> Number of Episodes per Iteration:", config.num_episodes)

        func_eval_budget = config.dim * 10000
        max_func_eval = config.swarm_size * config.num_episodes * config.obs_per_episode
        print(f"=== Function Evaluation Budget: {config.dim} dimensions x 10 000/dim = {func_eval_budget} Function Evaluations")
        print(f"=== Observations Per Episode: {config.num_swarm_obs_intervals} Number of Swarm Observation Intervals per Episode x {config.swarm_obs_interval_length} Number of Observations in each Interval  = {config.obs_per_episode}  Observations per Episode")
        print(f"=== Function Evaluation Allocation: {config.swarm_size} Swarm Size (# Particles) x {config.num_episodes} Episodes x {config.obs_per_episode} Observations per Episode  = {max_func_eval} Function Evaluations")
        print()

        if func_eval_budget != max_func_eval:
            raise ValueError("Maximum Function Evaluation budget does not match total allocations of function evaluations.")

        main.train()
    else:
        print("Re-Making Plots. Data will be reloaded from the results directory, ", config.results_dir, ", and re-plotted.")
        main.plot()

    if config.test:
        print(">> Testing mode. Number of Episodes to test: ", config.test_episodes, ". Will save test output to: ", config.test_step_results_path, "at the step, ", config.train_steps)
        main.test()
