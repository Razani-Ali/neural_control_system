import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')
from matplotlib.cm import get_cmap



def plot_open_loop(state_trajectories, outputs, reference, time, fig_size=(10, 8), continuous=False):
    fig, axes = plt.subplots(2, 1, figsize=fig_size)
    cmap = get_cmap('tab10')  # Use the 'tab10' colormap

    # Plot Output and Reference Trajectories
    for row in range(outputs.shape[0]):
        color = cmap(row % 10)  # Cycle through 'tab10' colors
        if continuous:
            axes[0].plot(time, reference[row, :], label=f"Reference {row + 1}", linestyle='--', color=color)
            axes[0].plot(time, outputs[row, :], label=f"Output {row + 1}", linestyle='-', color=color)
        else:
            axes[0].step(time, reference[row, :], label=f"Reference {row + 1}", linestyle='--', color=color)
            axes[0].step(time, outputs[row, :], label=f"Output {row + 1}", linestyle='-', color=color)

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Output Trajectory")
    axes[0].grid(True, color='silver')
    axes[0].legend()

    # Plot State Trajectories
    for row in range(state_trajectories.shape[0]):
        color = cmap(row % 10)  # Cycle through 'tab10' colors
        if continuous:
            axes[1].plot(time, state_trajectories[row, :], label=f"State {row + 1}", linestyle='-', color=color)
        else:
            axes[1].step(time, state_trajectories[row, :], label=f"State {row + 1}", linestyle='-', color=color)

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value")
    axes[1].set_title("State Trajectories")
    axes[1].grid(True, color='silver')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

#############################################################################################################################

def index_n_close_loop(loss_list, current_epoch, epochs, reference, output, time, state_trajectories,
                       control_effort, continuous=False, fig_size=(10, 8)):
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    cmap = get_cmap('tab10')  # Use the 'tab10' colormap

    # Plot Loss
    if (current_epoch is not None) and (epochs is not None):
        axes[0, 0].plot(range(1, len(loss_list) + 1), loss_list, label="Loss")
        axes[0, 0].scatter(len(loss_list), loss_list[-1], label="latest Loss", facecolors='none', edgecolors='blue')
    else:
        axes[0, 0].plot(0, loss_list, label="Loss")
        axes[0, 0].scatter(0, loss_list[-1], label="latest Loss", facecolors='none', edgecolors='blue')

    axes[0, 0].set_xlabel(f"epoch({current_epoch}/{epochs})")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"Loss, last epoch: {loss_list[-1]:.4f}")
    axes[0, 0].grid(True, color='silver', axis='y')
    axes[0, 0].legend()

    # Plot Reference Tracking
    for row in range(reference.shape[0]):
        color = cmap(row % 10)  # Cycle through 'tab10' colors
        if continuous:
            axes[0, 1].plot(time, reference[row, :], linestyle='--', label=f"Reference {row + 1}", color=color)
            axes[0, 1].plot(time, output[row, :], linestyle='-', label=f"Output {row + 1}", color=color)
        else:
            axes[0, 1].step(time, reference[row, :], linestyle='--', label=f"Reference {row + 1}", color=color)
            axes[0, 1].step(time, output[row, :], linestyle='-', label=f"Output {row + 1}", color=color)

    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_title("Reference Tracking")
    axes[0, 1].grid(True, color='silver')
    axes[0, 1].legend()

    # Plot State Trajectories
    for row in range(state_trajectories.shape[0]):
        color = cmap(row % 10)  # Cycle through 'tab10' colors
        if continuous:
            axes[1, 0].plot(time, state_trajectories[row, :], label=f"State {row + 1}", color=color)
        else:
            axes[1, 0].step(time, state_trajectories[row, :], label=f"State {row + 1}", color=color)

    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("State Trajectories")
    axes[1, 0].grid(True, color='silver')
    axes[1, 0].legend()

    # Plot Control Effort
    for row in range(control_effort.shape[0]):
        color = cmap(row % 10)  # Cycle through 'tab10' colors
        axes[1, 1].step(time, control_effort[row, :], label=f"Control {row + 1}", color=color)

    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("Control Effort")
    axes[1, 1].grid(True, color='silver')
    axes[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
