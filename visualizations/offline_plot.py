import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
plt.style.use('seaborn-v0_8-dark')


def plot_curve_fitting(
    actual_data: np.ndarray, 
    predicted_data: np.ndarray, 
    curve_labels: list, 
    title: str = 'Multiple Data Curves', 
    figsize: tuple = (12, 5)
) -> None:
    """
    Plots multiple actual vs predicted curves for comparison.

    Parameters:
    - actual_data (np.ndarray): 2D array where each row is a set of actual target values.
    - predicted_data (np.ndarray): 2D array where each row is a set of predicted values.
    - curve_labels (list): List of labels for each curve pair (actual, predicted).
    - title (str): Title for the plot. Defaults to 'Multiple Data Curves'.
    - figsize (tuple): Size of the figure for the plot, default is (12, 5).

    Returns:
    - None: Displays the plot.
    """

    # Ensure the dimensions of the inputs are consistent
    assert actual_data.shape == predicted_data.shape, "Actual and predicted data must have the same shape."
    assert len(curve_labels) == actual_data.shape[0], "Number of labels must match the number of curves."

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use a colormap to assign distinct colors to each pair of curves
    cmap = get_cmap('tab10')

    # Iterate through each curve and plot
    for i in range(actual_data.shape[0]):
        color = cmap(i % 10)  # Cycle through the colormap
        ax.plot(actual_data[i], color=color, label=f'{curve_labels[i]} Actual')
        ax.plot(predicted_data[i], color=color, linestyle=':', label=f'{curve_labels[i]} Predicted')

    # Add legend
    ax.legend(loc='upper right', fontsize='small')

    # Minor ticks for better granularity
    ax.minorticks_on()

    # Hide x-axis labels for clarity
    ax.tick_params(axis='x', labelbottom=False)

    # Determine the y-axis limits dynamically based on all data
    ymin = float(np.min([np.min(actual_data), np.min(predicted_data)]))
    ymax = float(np.max([np.max(actual_data), np.max(predicted_data)]))

    # Adjust ymin and ymax with scaling for better visual clarity
    if ymin < 0:
        ymin *= 1.1
    elif ymin == 0:
        ymin -= 0.1
    else:
        ymin *= 0.9

    if ymax < 0:
        ymax *= 0.9
    elif ymax == 0:
        ymax += 0.1
    else:
        ymax *= 1.1

    # Set y-limits
    ax.set_ylim(ymin, ymax)

    # Add grid lines for both x and y axes
    ax.grid(True, which='both', axis='both')

    # Set labels and title
    ax.set_ylabel('Values')
    ax.set_title(title)

    # Add an overall title and adjust layout
    fig.suptitle('Curve Fitting Plot for Multiple Data Curves')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()

#############################################################################################################################

def plot_regression(
    actual_data: np.ndarray, 
    predicted_data: np.ndarray, 
    figsize: tuple = (12, 5), 
    title: str = 'Data'
) -> None:
    """
    Plots a scatter plot of actual vs predicted values for a dataset, and fits a regression line.

    Parameters:
    - actual_data (np.ndarray): Actual target values.
    - predicted_data (np.ndarray): Predicted values by the model.
    - figsize (tuple): Size of the figure for the plot, default is (12, 5).
    - title (str): Title for the plot. Defaults to 'Data'.

    Returns:
    - None: Displays the scatter plot and regression line.
    """
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter plot of actual vs. predicted values
    ax.scatter(actual_data, predicted_data, color='blue', facecolors='none', label=title)

    # Calculate min and max values for the actual data
    min_value = min(actual_data)
    max_value = max(actual_data)

    # Fit a linear regression model to the data
    model = LinearRegression()
    model.fit(np.array(actual_data).reshape(-1, 1), predicted_data)
    a = model.coef_[0]  # Regression coefficient (slope)
    b = model.intercept_  # Intercept

    # Plot the reference line (perfect prediction)
    ax.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')

    # Plot the fitted regression line
    ax.plot([min_value, max_value], [a * min_value + b, a * max_value + b], 'k--', label='Fit Line')

    # Set axis labels and title
    ax.set_xlabel('Expected Values')
    ax.set_ylabel('Predicted Values')
    a = float(a)
    ax.set_title(f'{title}: Reg. Coeff = {a:.2f}')

    # Add legend
    ax.legend()

    # Turn on minor ticks and grid
    ax.minorticks_on()
    ax.grid(True, which='both')

    # Add an overall title and adjust layout
    fig.suptitle('Regression Plot')
    plt.tight_layout()

    # Show the plot
    plt.show()
