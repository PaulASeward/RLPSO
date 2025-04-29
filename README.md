# Reinforcement Learning with Particle Swarm Optimization (RL PSO) Project

## Features

- Utilization of both DQN and DDPG agents.
- Configurable PSO environment to test different functions and dimensions.
- Extensive logging and plotting utilities for detailed analysis of the agent's performance. 
- Multiple configurations to customize the PSO algorithm's behavior.

## Installation

Ensure you have Python 3.9 or newer installed. It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash 
pip install -r requirements.txt
```

If this virtual environment is already created and requirements installed into it, we can just activate to use it:

```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Running the Project

You can customize the training configuration by specifying arguments, for example:
    
```bash
python main.py --network_type DRQN --algorithm PSO --dim 30 --swarm_size 50 --func_num 6
```



## Project Structure
    ReinforcementLearningPSO/
        ├── agents/
        │   ├── agent.py
        │   ├── dqn_agent.py
        │   ├── drqn_agent.py
        │   ├── ddpg_agent.py
        |   ├── model_networks/
        │   |   ├── base_model.py
        │   |   ├── dqn_model.py
        │   |   ├── drqn_model.py
        │   |   ├── ddpg_model.py
        |   ├── utils/
        │   │   ├── experience_buffer.py
        │   │   └── policy.py
        ├── environment/
        │   ├── actions/
        │   ├── mock_env.py
        │   ├── gym_env_continuous.py 
        │   ├── gym_env_discrete.py
        ├── pso/
        │   ├── extdata/
        │   ├── standard_pso_results/
        │   ├── cec_benchmark_functions.py
        │   ├── pso_swarm.py
        │   ├── pso_multiswarm.py
        ├── model_networks/
        │   ├── base_model.py
        │   ├── dqn_model.py
        │   ├── drqn_model.py
        ├── utils/
        │   ├── logging_utils.py
        │   ├── plot_utils.py
        ├── config.py
        ├── main.py
        ├── README.md
        ├── requirements.txt
        ├── run_agent.sh

## Customizing Configuration
Edit the config.py file to set various parameters like network type, dimensions, swarm size, and more to suit different experimental needs.

## Agents
The agents/ directory contains the main execution agents of the project, including DQNAgent and DRQNAgent, which are built on the BaseAgent class. These agents are responsible for setting up the training environment, managing the experience replay buffer, and updating the models' weights. Users can specify the agent's policy and adjust various settings via the config.py file to tailor the learning process. Each agent script is designed to handle interactions with the PSO environment, collect and utilize experience data, and implement the chosen reinforcement learning strategies effectively.

## Model Networks
The model_networks/ directory houses the neural network architectures for the RL PSO Project, featuring the BaseModel class and its derivatives, DQNModel and DRQNModel. These models are configured to use different optimizers like Adam or RMSprop, and are integral for training on batches of experience data. Users can customize learning parameters such as learning rate and gamma in the config.py file. Each model compiles with a mean squared error loss function and supports detailed performance tracking and checkpoint saving for effective training session management and recovery. This setup allows for tailored neural network configurations to meet specific experimental needs within the project's reinforcement learning framework.

## Environment
The environment/ directory features the core simulation components for the RL PSO Project, central to which is the PSOEnv class. This class simulates a Particle Swarm Optimization (PSO) environment by interacting with the PSOSwarm class, which manages the dynamics of a particle swarm, including velocity, position updates, and fitness evaluations based on configurable parameters set in config.py. The environment defines the actions available to agents and translates these into modifications of particle behaviors or properties within the swarm, such as adjusting personal best thresholds or resetting particle states. Observations provided to the agents consist of concatenated arrays of velocities, relative fitness, and average replacement rates, which are critical for making informed decisions. Additionally, options for mocked or tracked location environments are available for experimenting or debugging purposes, providing flexibility in testing different scenarios or swarm behaviors.
