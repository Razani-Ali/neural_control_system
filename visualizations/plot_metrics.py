import numpy as np
from visualizations.online_plot import loss_n_ctr_perf, loss_plot, fitting_loss_n_ctr_perf, fitting_loss

def plot_metrics(number_of_epochs: int, 
    current_epoch: int,
    end_time: float, 
    loss: np.ndarray, 
    obs_iden_traj: np.ndarray, 
    obs_iden_out: np.ndarray,
    ctrl_perf_ind: np.ndarray = None,
    trajectory: np.ndarray = None, 
    **kwargs
) -> None:

    # Retrieve the user's option to plot the loss curve (defaults to True)
    plot_loss = kwargs.get('plot_loss', True)
    plot_perf = kwargs.get('plot_perf', True)

    fig_size = kwargs.get('fig_size', (8, 6))

    # If plotting loss is enabled
    if plot_loss and plot_perf:

        plot_fitting = kwargs.get('plot_fitting', False)  # Plot fitting (regression) option

        # Case 1: Both fitting and loss plotting are enabled
        if plot_fitting:
            # Call regression fitting plot
            fitting_loss_n_ctr_perf(number_of_epochs, current_epoch, end_time, loss,
                ctrl_perf_ind, obs_iden_traj, obs_iden_out, trajectory, figure_size = fig_size)
            
        # Case 2: Only plot loss
        else:
            # Fallback to basic loss plot if no other options are valid
            loss_n_ctr_perf(number_of_epochs, current_epoch, loss, ctrl_perf_ind, figure_size = fig_size)

    elif plot_loss:
        
        plot_fitting = kwargs.get('plot_fitting', False)  # Plot fitting (regression) option

        # Case 3: Both fitting and loss plotting are enabled
        if plot_fitting:
            # Call regression fitting plot
            fitting_loss(number_of_epochs, current_epoch, end_time, 
                loss, obs_iden_traj, obs_iden_out, figure_size = fig_size)
            
        # Case 4: Only plot loss
        else:
            # Fallback to basic loss plot if no other options are valid
            loss_plot(number_of_epochs, current_epoch, loss, figure_size = fig_size)
