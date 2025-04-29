import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import os


def plot_standard_results(config):
    plot_data_over_iterations(config.average_returns_path, 'Average Return', 'Iteration', config.eval_interval)
    plot_data_over_iterations(config.fitness_path, 'Average Fitness', 'Iteration', config.eval_interval)
    plot_data_over_iterations(config.loss_file, 'Average Loss', 'Iteration', config.log_interval)
    plot_two_datasets_over_iterations(config.average_returns_path, 'Average Return', config.epsilon_values_path, 'Epsilon Of Policy', 'Iteration', config.eval_interval)
    plot_two_datasets_over_iterations(config.fitness_path, 'Average Fitness', config.epsilon_values_path, 'Epsilon Of Policy', 'Iteration', config.eval_interval)


def plot_discrete_actions(config):
    if config.swarm_algorithm == "PMSO":
        plot_actions_over_iteration_intervals_for_multiple_swarms(config.interval_actions_counts_path, config.fitness_path,
                                              'Iteration Intervals', 'Action Count',
                                              'Action Distribution Over Iteration Intervals',
                                              config.iteration_intervals,
                                              config.label_iterations_intervals,
                                              config.actions_descriptions, num_subswarms=config.num_sub_swarms)
        plot_actions_with_values_over_iteration_intervals_for_multiple_swarms(config.action_counts_path,
                                                                              config.action_values_path,
                                                                              standard_pso_values_path=config.standard_pso_path,
                                                                              function_min_value=config.fDeltas[
                                                                                  config.func_num - 1],
                                                                              num_actions=config.num_actions,
                                                                              action_names=config.actions_descriptions,
                                                                              num_subswarms=config.num_sub_swarms)
    else:
        plot_actions_over_iteration_intervals(config.interval_actions_counts_path, config.fitness_path,
                                              'Iteration Intervals', 'Action Count',
                                              'Action Distribution Over Iteration Intervals',
                                              config.iteration_intervals,
                                              config.label_iterations_intervals,
                                              config.actions_descriptions)
        plot_actions_with_values_over_iteration_intervals(config.action_counts_path,
                                                          config.action_values_path,
                                                          standard_pso_values_path=config.standard_pso_path,
                                                          function_min_value=config.fDeltas[
                                                              config.func_num - 1],
                                                          num_actions=config.num_actions,
                                                          action_names=config.actions_descriptions)



def plot_continuous_actions(config):
    plot_data_over_iterations(config.actor_loss_file, 'Average Actor Loss', 'Iteration', config.log_interval)
    plot_data_over_iterations(config.critic_loss_file, 'Average Critic Loss', 'Iteration', config.log_interval)


    if config.swarm_algorithm == "PMSO":
        plot_average_continuous_actions_for_multiple_swarms(config.continuous_action_history_path,
                                                            config.action_values_path,
                                                            standard_pso_values_path=config.standard_pso_path,
                                                            function_min_value=config.fDeltas[
                                                                config.func_num - 1],
                                                            action_dimensions=config.action_dimensions,
                                                            action_names=config.actions_descriptions,
                                                            practical_action_low_limit = config.practical_action_low_limit,
                                                            practical_action_high_limit = config.practical_action_high_limit,
                                                            num_intervals=9)
    else:
        plot_average_continuous_actions_for_single_swarm(config.continuous_action_history_path,
                                                         config.action_values_path,
                                                         standard_pso_values_path=config.standard_pso_path,
                                                         function_min_value=config.fDeltas[
                                                             config.func_num - 1],
                                                         action_dimensions=config.action_dimensions,
                                                         action_names=config.actions_descriptions,
                                                         practical_action_low_limit=config.practical_action_low_limit,
                                                         practical_action_high_limit=config.practical_action_high_limit,
                                                         num_intervals=15)

def plot_data_over_iterations(file_name, y_label, x_label, iteration_interval_scale):
    y_data = np.genfromtxt(file_name, delimiter=',')
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    iterations = range(0, len(y_data) * iteration_interval_scale, iteration_interval_scale)
    plt.plot(iterations, y_data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(output_file_name, dpi='figure', format="png", metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    plt.close()


def plot_two_datasets_over_iterations(file_name, y_label, file_name2, y_label2, x_label, iteration_interval_scale):
    y_data = np.genfromtxt(file_name, delimiter=',')
    y_data2 = np.genfromtxt(file_name2, delimiter=',')
    iterations_y_data = range(0, len(y_data) * iteration_interval_scale, iteration_interval_scale)
    iterations_y_data2 = range(0, len(y_data2))

    output_file_name = os.path.splitext(file_name)[0] + '_dual_plot.png'

    fig, ax1 = plt.subplots()

    # Plot the first dataset on the primary y-axis
    line1, = ax1.plot(iterations_y_data, y_data, label=y_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    # Create a second y-axis for the second dataset
    ax2 = ax1.twinx()
    line2, = ax2.plot(iterations_y_data2, y_data2, 'r-', label=y_label2)
    ax2.set_ylabel(y_label2)

    # Combine both lines into a single legend
    lines = [line1, line2]
    labels = [y_label, y_label2]
    ax1.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_actions_over_iteration_intervals(file_name, relative_fitness, x_label, y_label, title, iteration_intervals, label_iteration_intervals, action_names):
    action_counts = np.genfromtxt(file_name, delimiter=',')
    relative_fitness_values = np.genfromtxt(relative_fitness, delimiter=',')
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    num_actions = action_counts.shape[1]
    print("Number of Actions Being Plotted: ", num_actions)
    print("Number of Iteration Intervals: ", len(iteration_intervals))
    print("Number of Labels for Iteration Intervals: ", len(label_iteration_intervals))
    print("Number of Action Counts Rows: ", action_counts.shape[0])

    if len(iteration_intervals) != action_counts.shape[0]:
        print(f"Warning: Mismatch between iteration intervals ({len(iteration_intervals)}) and action counts rows ({action_counts.shape[0]}). Adjusting to the minimum size.")
        min_length = min(len(iteration_intervals), action_counts.shape[0])
        iteration_intervals = iteration_intervals[:min_length]
        action_counts = action_counts[:min_length, :]
        label_iteration_intervals = label_iteration_intervals[:min_length]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create a bar plot for each action
    bar_width = 0.8 * (iteration_intervals[1] - iteration_intervals[0])
    bottom = np.zeros(len(iteration_intervals))

    for action in range(num_actions):
        ax1.bar(iteration_intervals, action_counts[:, action], bottom=bottom, width=bar_width, label=action_names[action])
        bottom += action_counts[:, action]

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Action Count')
    ax1.set_title(title)

    # Create a second y-axis for the relative fitness values
    ax2 = ax1.twinx()
    ax2.plot(iteration_intervals, relative_fitness_values, label='Average Relative Fitness', color='black', marker='o')
    ax2.set_ylabel('Average Relative Fitness')

    # Set x-axis ticks and labels with rotated tick labels
    ax1.set_xticks(label_iteration_intervals)
    ax1.set_xticklabels([str(i) for i in label_iteration_intervals], rotation=45, ha="right", fontsize=8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()

def adjust_color(color, fraction: float):
    """
    Lightens or darkens a base color by interpolating between white and the base color.
    fraction=0.0 => white
    fraction=1.0 => original color
    """
    # Convert to RGB tuple in [0, 1]
    base = mcolors.to_rgb(color)
    # Interpolate towards white
    white = (1, 1, 1)
    new_color = tuple((1 - fraction) * w + fraction * b for w, b in zip(white, base))
    return new_color

def get_action_colors(action_names, num_subswarms, base_cmap='viridis'):
    """
    Returns a list of colors so that:
    - action_names[0] is "Do Nothing" => grey
    - subsequent actions come in groups (one group per subswarm)
    each group is assigned a distinct base color from base_cmap,
    and the actions within that group are light-to-dark variants of that base color.
    """
    # First color is always grey for Doing nothing
    colors = ['grey']
    total_actions = len(action_names)
    # Number of subswarm actions = total_actions - 1 (excluding "Do Nothing")
    # Suppose you know each subswarm has 4 actions:
    actions_per_subswarm = (total_actions - 1) // num_subswarms

    # Fetch the colormap
    cmap = plt.cm.get_cmap(base_cmap, num_subswarms)

    # Build colors for each subswarm
    index = 1  # start after "Do Nothing"
    for subswarm_i in range(num_subswarms):
        base_color = cmap(subswarm_i)
        for action_i in range(actions_per_subswarm):
            # fraction for how "dark" to make this action
            fraction = 0.5 + 0.5 * (action_i / max(1, actions_per_subswarm - 1))
            # lighten or darken around 50%-100% of the base color
            shade = adjust_color(base_color, fraction)
            colors.append(shade)
            index += 1

    return colors

def plot_actions_over_iteration_intervals_for_multiple_swarms(file_name, relative_fitness, x_label, y_label, title, iteration_intervals, label_iteration_intervals, action_names, num_subswarms):
    action_counts = np.genfromtxt(file_name, delimiter=',')
    relative_fitness_values = np.genfromtxt(relative_fitness, delimiter=',')
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    num_actions = action_counts.shape[1]
    colors = get_action_colors(action_names, num_subswarms=num_subswarms, base_cmap='viridis')
    print("Number of Actions Being Plotted: ", num_actions)
    print("Number of Iteration Intervals: ", len(iteration_intervals))
    print("Number of Labels for Iteration Intervals: ", len(label_iteration_intervals))
    print("Number of Action Counts Rows: ", action_counts.shape[0])

    if len(iteration_intervals) != action_counts.shape[0]:
        print(f"Warning: Mismatch between iteration intervals ({len(iteration_intervals)}) and action counts rows ({action_counts.shape[0]}). Adjusting to the minimum size.")
        min_length = min(len(iteration_intervals), action_counts.shape[0])
        iteration_intervals = iteration_intervals[:min_length]
        action_counts = action_counts[:min_length, :]
        label_iteration_intervals = label_iteration_intervals[:min_length]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create a bar plot for each action
    bar_width = 0.8 * (iteration_intervals[1] - iteration_intervals[0])
    bottom = np.zeros(len(iteration_intervals))

    for action in range(num_actions):
        ax1.bar(iteration_intervals, action_counts[:, action], bottom=bottom, width=bar_width, label=action_names[action],
                color=colors[action] if action < len(colors) else None
                )
        bottom += action_counts[:, action]

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Action Count')
    ax1.set_title(title)

    # Create a second y-axis for the relative fitness values
    ax2 = ax1.twinx()
    ax2.plot(iteration_intervals, relative_fitness_values, label='Average Relative Fitness', color='black', marker='o')
    ax2.set_ylabel('Average Relative Fitness')

    # Set x-axis ticks and labels with rotated tick labels
    ax1.set_xticks(label_iteration_intervals)
    ax1.set_xticklabels([str(i) for i in label_iteration_intervals], rotation=45, ha="right", fontsize=8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.55),  # 0.5 is center x, 1.15 is above the axes
        ncol=2,  # spread legend items into 2 columns
        frameon=True  # give legend a border if desired
    )
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_average_continuous_actions_for_single_swarm(continuous_action_history_path, actions_values_path, standard_pso_values_path, function_min_value, action_dimensions, action_names, practical_action_low_limit, practical_action_high_limit, num_intervals=9):
    output_file_name = os.path.splitext(continuous_action_history_path)[0] + '_single_swarm.png'
    action_counts = np.load(continuous_action_history_path)
    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    # Fill in the action values with zeros to match the length of the action counts
    action_values = np.pad(action_values, ((0, len(action_counts) - len(action_values)), (0, 0)), 'constant')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(action_dimensions)]
    fig.legend(legend_handles, action_names[:action_dimensions], loc='upper right', title="Actions")

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    low_limit = np.array([x if x is not None else -np.inf for x in practical_action_low_limit], dtype=float)
    high_limit = np.array([x if x is not None else np.inf for x in practical_action_high_limit], dtype=float)
    action_counts = np.clip(action_counts, low_limit, high_limit)

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        mean_action_counts = np.mean(interval_data, axis=0)
        std_action_counts = np.std(interval_data, axis=0)

        # Plot each action dimension with a shaded area for the std deviation
        for j in range(action_dimensions):
            mean_counts = mean_action_counts[:, j]
            std_dev_counts = std_action_counts[:, j]
            ax.plot(x_values, mean_counts, color=f'C{j}', label=action_names[j])
            ax.fill_between(x_values, mean_counts - std_dev_counts, mean_counts + std_dev_counts,
                            color=f'C{j}', alpha=0.3)

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Continuous Action Average")
        ax.set_title(f'Continuous Action Averages - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_average_continuous_actions_for_multiple_swarms(continuous_action_history_path, actions_values_path, standard_pso_values_path, function_min_value, action_dimensions, action_names, practical_action_low_limit, practical_action_high_limit, num_intervals=9, separate_plot_for_interval=6):
    output_file_name = os.path.splitext(continuous_action_history_path)[0] + '_multiple_swarms.png'
    action_counts = np.load(continuous_action_history_path)

    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    # Fill in the action values with zeros to match the length of the action counts
    action_values = np.pad(action_values, ((0, len(action_counts) - len(action_values)), (0, 0)), 'constant')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(action_dimensions)]
    fig.legend(legend_handles, action_names[:action_dimensions], loc='upper right', title="Actions")

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    # Clamp the actions to practical limits for plotting.
    low_limit = np.array([x if x is not None else -np.inf for x in practical_action_low_limit], dtype=float)
    high_limit = np.array([x if x is not None else np.inf for x in practical_action_high_limit], dtype=float)
    action_counts = np.clip(action_counts, low_limit, high_limit)

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        mean_action_counts = np.mean(interval_data, axis=0)
        std_action_counts = np.std(interval_data, axis=0)

        # Plot each action dimension with a shaded area for the std deviation
        for j in range(action_dimensions):
            mean_counts = mean_action_counts[:, j]
            std_dev_counts = std_action_counts[:, j]

            # # Scale even-indexed action dimensions (0, 2, 4, ...) from [0, 200] to [0, 2]
            # if j % 2 == 0:
            #     mean_counts = mean_counts / 100
            #     std_dev_counts = std_dev_counts / 100

            ax.plot(x_values, mean_counts, color=f'C{j}', label=action_names[j])
            ax.fill_between(x_values, mean_counts - std_dev_counts, mean_counts + std_dev_counts, color=f'C{j}', alpha=0.3)

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Continuous Action Average")
        ax.set_title(f'Continuous Action Averages - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

        # Make a subplot into a new figure
        if separate_plot_for_interval is not None and i == separate_plot_for_interval:
            # Create a new figure for the separate plot
            fig_single, ax_single = plt.subplots(figsize=(10, 6))

            cmap = cm.get_cmap('turbo')  # Or 'plasma', 'turbo', etc.
            colors = [mcolors.to_hex(cmap(i / int(action_dimensions-1))) for i in range(action_dimensions)]
            # colors = [mcolors.to_hex((0.3 * (1 - j / max(action_dimensions - 1, 1)), 0.5 * (1 - j / max(action_dimensions - 1, 1)), 1.0)) for j in range(action_dimensions)]
            # legend_handles = [Patch(facecolor=f'C{i}') for i in range(action_dimensions)]
            legend_handles = [Patch(facecolor=colors[i]) for i in range(action_dimensions)]

            # Re-plot the action dimensions
            for j in range(action_dimensions):
                mean_counts = mean_action_counts[:, j]
                std_dev_counts = std_action_counts[:, j]
                #
                # # Scale even-indexed action dimensions (0, 2, 4, ...) from [0, 200] to [0, 2]
                # if j % 2 == 0:
                #     mean_counts = mean_counts / 100
                #     std_dev_counts = std_dev_counts / 100

                ax_single.plot(x_values, mean_counts, color=colors[j], label=action_names[j])
                ax_single.fill_between(x_values, mean_counts - std_dev_counts, mean_counts + std_dev_counts,
                                       color=colors[j], alpha=0.3)
                # ax_single.plot(x_values, mean_counts, color=f'C{j}', label=action_names[j])
                # ax_single.fill_between(x_values, mean_counts - std_dev_counts, mean_counts + std_dev_counts,
                #                        color=f'C{j}', alpha=0.3)

            ax_single.set_xlabel("Episode Number")
            ax_single.set_ylabel("Continuous Action Average")
            ax_single.set_title(f'Continuous Action Converged Policy')
            ax_single.set_xticks(x_values)



            # legends
            fig_single.legend(legend_handles, action_names[:action_dimensions], loc='upper center', title="Actions", ncol=3, bbox_to_anchor=(0.5, 1.11))
            # Save the individual subplot
            single_output_file_path = os.path.splitext(continuous_action_history_path)[0] + f'_interval_{i + 1}.png'
            fig_single.tight_layout()
            fig_single.savefig(single_output_file_path, dpi='figure', format="png", bbox_inches='tight')
            plt.close(fig_single)

        # if separate_plot_for_interval is not None and i == separate_plot_for_interval:
        #     fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 6))
        #
        #     # Extract binary actions (1 if > 0.5, else 0)
        #     binary_actions = (interval_data > 0.5).astype(int)  # shape: [num_steps, action_dims]
        #     heatmap_data = np.sum(binary_actions, axis=0).T  # Shape: (25, 20) for (y, x)
        #     heatmap = ax_heatmap.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
        #
        #     ax_heatmap.set_xlabel("Timestep")
        #     ax_heatmap.set_ylabel("Action Dimension")
        #     ax_heatmap.set_title(f'Reset Counts (Action Value in Dimension >0.5)')
        #     ax_heatmap.set_xticks(np.arange(20))
        #     ax_heatmap.set_yticks(np.arange(25))
        #     ax_heatmap.set_yticklabels([f"{name}" for name in action_names[:25]])  # if needed
        #
        #     # Color bar
        #     cbar = plt.colorbar(heatmap, ax=ax_heatmap)
        #     cbar.set_label("Reset Count")
        #
        #     # Save heatmap figure
        #     heatmap_file = os.path.splitext(continuous_action_history_path)[0] + f'_interval_{i + 1}_heatmap.png'
        #     fig_heatmap.tight_layout()
        #     fig_heatmap.savefig(heatmap_file, dpi='figure', format="png", bbox_inches='tight')
        #     plt.close(fig_heatmap)

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()

def plot_actions_with_values_over_iteration_intervals_for_multiple_swarms(actions_counts_path, actions_values_path, standard_pso_values_path, function_min_value, num_actions, action_names, num_subswarms, num_intervals=9):
    output_file_name = os.path.splitext(actions_counts_path)[0] + '.png'
    action_counts = np.genfromtxt(actions_counts_path, delimiter=',')
    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)
    colors = get_action_colors(action_names, num_subswarms=num_subswarms, base_cmap='viridis')


    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=colors[i]) for i in range(num_actions)]
    fig.legend(legend_handles, action_names[:num_actions], loc='upper center', title="Actions",
               bbox_to_anchor=(0.5, 1.15),  # 0.5 is center x, 1.15 is above the axes
               ncol=2,  # spread legend items into 2 columns
               frameon=True)

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        bottom = np.zeros(num_episodes)
        for action_num in range(num_actions):
            action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in range(num_episodes)]
            ax.bar(x_values, action_occurrences, bottom=bottom, label=action_names[action_num],
                   color=colors[action_num] if action_num < len(colors) else None)
            bottom += action_occurrences

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Action Count")
        ax.set_title(f'Action Counts - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

        # Make a subplot into a new figure
        if i == 6:
            # Define hatches
            hatch_patterns = ['', '///', '\\\\\\', 'xxx', '---', '///', '\\\\\\', 'xxx', '---']
            # 0 -> no hatch, 1->///, 2->\\\, 3->xxx, 4->---, then repeat for 5–8

            # Create a new figure for the separate plot
            fig_single, ax_single = plt.subplots(figsize=(10, 6))

            bottom = np.zeros(num_episodes)
            for action_num in range(num_actions):
                action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in
                                      range(num_episodes)]
                ax_single.bar(
                    x_values,
                    action_occurrences,
                    bottom=bottom,
                    label=action_names[action_num],
                    hatch=hatch_patterns[action_num],
                    color=colors[action_num] if action_num < len(colors) else None
                )
                bottom += action_occurrences

            ax_single.set_xlabel("Episode Number")
            ax_single.set_ylabel("Action Count")
            ax_single.set_title(f'Discrete Action Converged Policy')
            ax_single.set_xticks(x_values)

            # Add line graph overlay
            ax2_single = ax_single.twinx()
            ax2_single.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
            ax2_single.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
            ax2_single.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
            ax2_single.set_ylim(min_line_value, max_line_value)
            ax2_single.legend(loc='upper left')

            legend_handles = [
                Patch(
                    facecolor=colors[i],
                    hatch=hatch_patterns[i],
                    edgecolor='black'
                ) for i in range(num_actions)
            ]

            # legends
            fig_single.legend(legend_handles, action_names[:num_actions],  loc='upper center', title="Actions",
                              ncol=3, bbox_to_anchor=(0.5, 1.15))
            # Save the individual subplot
            single_output_file_path = os.path.splitext(actions_counts_path)[0] + f'_interval_{i + 1}.png'
            fig_single.tight_layout()
            fig_single.savefig(single_output_file_path, dpi='figure', format="png", bbox_inches='tight')
            plt.close(fig_single)

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_actions_with_values_over_iteration_intervals(actions_counts_path, actions_values_path, standard_pso_values_path, function_min_value, num_actions, action_names, num_intervals=9):
    output_file_name = os.path.splitext(actions_counts_path)[0] + '.png'
    action_counts = np.genfromtxt(actions_counts_path, delimiter=',')
    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Define hatches
    hatch_patterns = ['', '///', '\\\\\\', 'xxx', '---', '///', '\\\\\\', 'xxx', '---']
    # 0 -> no hatch, 1->///, 2->\\\, 3->xxx, 4->---, then repeat for 5–8

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(num_actions)]
    fig.legend(legend_handles, action_names[:num_actions], loc='upper right', title="Actions")

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        bottom = np.zeros(num_episodes)
        for action_num in range(num_actions):
            action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in range(num_episodes)]
            ax.bar(x_values, action_occurrences, bottom=bottom, label=action_names[action_num])
            bottom += action_occurrences

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Action Count")
        ax.set_title(f'Action Counts - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

        # Make a subplot into a new figure
        if i == 6:
            # Create a new figure for the separate plot
            fig_single, ax_single = plt.subplots(figsize=(10, 6))

            bottom = np.zeros(num_episodes)
            for action_num in range(num_actions):
                action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in
                                      range(num_episodes)]
                ax_single.bar(
                    x_values,
                    action_occurrences,
                    bottom=bottom,
                    label=action_names[action_num],
                    hatch=hatch_patterns[action_num]
                )
                bottom += action_occurrences


            ax_single.set_xlabel("Episode Number")
            ax_single.set_ylabel("Action Count")
            ax_single.set_title(f'Discrete Action Converged Policy')
            ax_single.set_xticks(x_values)

            # Add line graph overlay
            ax2_single = ax_single.twinx()
            ax2_single.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
            ax2_single.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
            ax2_single.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
            ax2_single.set_ylim(min_line_value, max_line_value)
            ax2_single.legend(loc='upper left')

            # legends
            fig_single.legend(legend_handles, action_names[:num_actions], loc='upper center', title="Actions", ncol=3, bbox_to_anchor=(0.5, 1.15))
            # Save the individual subplot
            single_output_file_path = os.path.splitext(actions_counts_path)[0] + f'_interval_{i + 1}.png'
            fig_single.tight_layout()
            fig_single.savefig(single_output_file_path, dpi='figure', format="png", bbox_inches='tight')
            plt.close(fig_single)

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()
