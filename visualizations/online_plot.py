import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
plt.style.use('seaborn-v0_8-dark')

def loss_n_ctr_perf(
    number_of_epochs: int, 
    current_epoch: int, 
    loss: np.ndarray, 
    ctrl_perf_ind: np.ndarray, 
    figure_size: tuple = (12, 5)
) -> None:

    # Create a figure with two subplots, horizontally aligned
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # Plot training loss on the first subplot with a logarithmic y-scale
    ax1.plot(range(1, current_epoch + 1), loss, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax1.set_ylabel('Loss')  # Label the y-axis
    ax1.set_title(f'Obvervation or Identification Loss,\nlast epoch: {loss[-1]:.5f}')  # Title showing the last training loss
    ax1.yaxis.grid(True, which='minor')  # Add grid lines for the y-axis (minor scale for better granularity)
    ax1.xaxis.grid(False)  # Turn off x-axis grid
    
    # Plot validation loss on the second subplot with a logarithmic y-scale
    ax2.plot(range(1, current_epoch + 1), ctrl_perf_ind, color='orange')
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax2.set_ylabel('Index')  # Label the y-axis
    ax2.set_title(f'Control Performance Index,\nlast epoch: {ctrl_perf_ind[-1]:.5f}')  # Title showing the last validation loss
    ax2.yaxis.grid(True, which='minor')  # Add grid lines for the y-axis (minor scale for better granularity)
    ax2.xaxis.grid(False)  # Turn off x-axis grid
    
    # Set an overarching title for both plots
    fig.suptitle('Live Loss and Performance Plots')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plot
    plt.show()

#############################################################################################################################

def loss_plot(
    number_of_epochs: int, 
    current_epoch: int, 
    loss: np.ndarray, 
    figure_size: tuple = (12, 5)
) -> None:
    
    # Create a figure with two subplots, horizontally aligned
    fig, ax1 = plt.subplots(1, 1, figsize=figure_size)
    
    # Plot training loss on the first subplot with a logarithmic y-scale
    ax1.plot(range(1, current_epoch + 1), loss, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax1.set_ylabel('Loss')  # Label the y-axis
    ax1.set_title(f'Observation or Identification Loss, last epoch: {loss[-1]:.5f}')  # Title showing the last training loss
    ax1.yaxis.grid(True, which='minor')  # Add grid lines for the y-axis (minor scale for better granularity)
    ax1.xaxis.grid(False)  # Turn off x-axis grid
    
    # Set an overarching title for both plots
    fig.suptitle('Live Loss Plots')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plot
    plt.show()

#############################################################################################################################

def fitting_loss_n_ctr_perf(
    number_of_epochs: int, 
    current_epoch: int,
    end_time: float, 
    loss: np.ndarray, 
    ctrl_perf_ind: np.ndarray, 
    obs_iden_traj: np.ndarray, 
    obs_iden_out: np.ndarray, 
    trajectory: np.ndarray, 
    figure_size: tuple = (10, 6)
) -> None:
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Extract individual axes for easy reference
    ax1, ax2 = axes[0]
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(np.arange(1, current_epoch + 1), loss, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis with epoch info
    ax1.set_ylabel('Loss')  # Label y-axis as 'Loss'
    ax1.set_title(f'Observation or Identification Loss,\nlast epoch: {loss[-1]:.5f}')  # Title showing last train loss value
    ax1.yaxis.grid(True, which='minor')  # Enable minor grid lines for clarity on the y-axis
    ax1.xaxis.grid(False)  # Disable x-axis grid
    
    # Plot validation loss with a logarithmic y-scale
    ax2.plot(np.arange(1, current_epoch + 1), ctrl_perf_ind, color='orange')
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label x-axis with epoch info
    ax2.set_ylabel('Loss')  # Label y-axis as 'Loss'
    ax2.set_title(f'Control Performance Index,\nlast epoch: {ctrl_perf_ind[-1]:.5f}')  # Title showing last validation loss value
    ax2.yaxis.grid(True, which='minor')  # Enable minor grid lines for clarity on the y-axis
    ax2.xaxis.grid(False)  # Disable x-axis grid
    
    # Extract the second row of axes for actual vs predicted curves
    ax3, ax4 = axes[1]

    # Time axis
    time_range = np.linspace(0, end_time + 1e-3, obs_iden_traj.shape[1])
    
    # Use a colormap to assign colors to each trajectory
    cmap = get_cmap('tab10')  # A colormap with 10 distinct colors
    num_trajectories = obs_iden_traj.shape[0]  # Number of trajectories
    
    for i in range(num_trajectories):
        color = cmap(i % 10)  # Cycle through the colormap for unique colors
        
        # Plot observation trajectories and their outputs
        ax3.plot(time_range, obs_iden_traj[i], color=color, label=f'Traj {i + 1} Actual')
        ax3.plot(time_range, obs_iden_out[i], color=color, linestyle=':', label=f'Traj {i + 1} Predicted')
    
    ax3.set_title('Observation or Identification Curve Fitting')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend(loc='upper right', fontsize='small')  # Add a legend for clarity
    ax3.grid(True, which='both', axis='both', color='silver')
    
    # Plot the overall trajectory in ax4
    for i in range(trajectory.shape[0]):
        color = cmap(i % 10)  # Use the same color scheme
        ax4.plot(time_range, trajectory[i], color=color, label=f'Traj {i + 1}')
    
    ax4.set_title('Trajectories')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')
    ax4.grid(True, which='both', axis='both', color='silver')
    ax4.legend(loc='upper right', fontsize='small')  # Add a legend for clarity
    
    # Set an overarching title for the figure
    fig.suptitle('Live Loss, Performance and Curve Fitting Plots')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()

#############################################################################################################################

def fitting_loss(
    number_of_epochs: int, 
    current_epoch: int,
    end_time: float, 
    loss: np.ndarray, 
    obs_iden_traj: np.ndarray, 
    obs_iden_out: np.ndarray, 
    figure_size: tuple = (10, 6)
) -> None:
    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(2, 1, figsize=figure_size)
    
    # Extract individual axes for easy reference
    ax1, ax2 = axes
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(np.arange(1, current_epoch + 1), loss, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis with epoch info
    ax1.set_ylabel('Loss')  # Label y-axis as 'Loss'
    ax1.set_title(f'Observation or Identification Loss,\nlast epoch: {loss[-1]:.5f}')  # Title showing last train loss value
    ax1.yaxis.grid(True, which='minor')  # Enable minor grid lines for clarity on the y-axis
    ax1.xaxis.grid(False)  # Disable x-axis grid
    
    # Time axis
    time_range = np.linspace(0, end_time + 1e-3, obs_iden_traj.shape[1])
    
    # Use a colormap to assign colors to each trajectory
    cmap = get_cmap('tab10')  # A colormap with 10 distinct colors
    num_trajectories = obs_iden_traj.shape[0]  # Number of trajectories
    
    # Plot observation or identification trajectories
    for i in range(num_trajectories):
        color = cmap(i % 10)  # Cycle through the colormap for unique colors
        
        # Plot observation trajectories and their outputs
        ax2.plot(time_range, obs_iden_traj[i], color=color, label=f'Traj {i + 1} Actual')
        ax2.plot(time_range, obs_iden_out[i], color=color, linestyle=':', label=f'Traj {i + 1} Predicted')
    
    ax2.set_title('Observation or Identification Curve Fitting')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend(loc='upper right', fontsize='small')  # Add a legend for clarity
    ax2.grid(True, which='both', axis='both', color='silver')
    
    # Set an overarching title for the figure
    fig.suptitle('Live Loss and Curve Fitting Plots')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()
