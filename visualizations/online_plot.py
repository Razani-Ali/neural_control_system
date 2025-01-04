import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')


#############################################################################################################################

def train_loss(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    figure_size: tuple = (8, 5)
) -> None:
    """
    Plots the training loss over epochs using logarithmic scale for both axes.
    
    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch to date.
    - loss_train (np.ndarray): Array of training loss values per epoch.
    - figure_size (tuple): Size of the plot, default is (8, 5).

    Returns:
    - None: Displays the plot.
    """
    
    # Create a figure for the training loss plot
    fig, ax1 = plt.subplots(figsize=figure_size)
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax1.set_ylabel('Loss')  # Label the y-axis
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')  # Title showing the last training loss
    ax1.yaxis.grid(True, which='minor')  # Add grid lines for the y-axis (minor scale for better granularity)
    ax1.xaxis.grid(False)  # Turn off x-axis grid
    
    # Set y-limits for the plot
    ymin = float(min(loss_train)) * 0.9  # Set lower bound to 90% of the smallest value
    ymax = float(max(loss_train)) * 1.1  # Set upper bound to 110% of the largest value
    ax1.set_ylim(ymin, ymax)  # Apply the limits to the plot
    
    # Set a title for the figure
    fig.suptitle('Live Loss Plot')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plot
    plt.show()

#############################################################################################################################

def train_loss_fitting(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    actual_train: np.ndarray, 
    predicted_train: np.ndarray, 
    figure_size: tuple = (8, 6)
) -> None:
    """
    Plots training loss and the actual vs predicted curves for the training dataset.
    
    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch to date.
    - loss_train (np.ndarray): Array of training loss values per epoch.
    - actual_train (np.ndarray): Actual output values for the training set.
    - predicted_train (np.ndarray): Predicted output values for the training set.
    - figure_size (tuple): Size of the plot, default is (8, 6).
    
    Returns:
    - None: Displays the loss and fitting plots.
    """
    # Ensure `actual_train` and `predicted_train` are 2D arrays for consistent handling
    actual_train = np.atleast_2d(actual_train)
    predicted_train = np.atleast_2d(predicted_train)
    
    # Check if the dimensions match
    if actual_train.shape != predicted_train.shape:
        raise ValueError("Shape mismatch: `actual_train` and `predicted_train` must have the same shape.")
    
    num_columns = actual_train.shape[1]  # Number of columns to plot
    colors = plt.cm.get_cmap('tab10', num_columns)  # Get a colormap with enough distinct colors
    
    # Create a 2-row subplot grid
    fig, axes = plt.subplots(2, 1, figsize=figure_size)
    
    # Extract individual axes for easy reference
    ax1, ax2 = axes
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(np.arange(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis with epoch info
    ax1.set_ylabel('Loss')  # Label y-axis as 'Loss'
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')  # Title showing last train loss value
    ax1.yaxis.grid(True, which='minor')  # Enable minor grid lines for clarity on the y-axis
    ax1.xaxis.grid(False)  # Disable x-axis grid
    
    # Set y-axis limits for training loss plot
    ymin = float(min(loss_train)) * 0.9  # 90% of the minimum loss
    ymax = float(max(loss_train)) * 1.1  # 110% of the maximum loss
    ax1.set_ylim(ymin, ymax)
    
    # Plot each column of actual and predicted values with different colors
    for col_idx in range(num_columns):
        color = colors(col_idx)  # Get a color for this column
        ax2.plot(actual_train[:, col_idx], label=f'Actual Train {col_idx + 1}', color=color, linestyle='-')
        ax2.plot(predicted_train[:, col_idx], label=f'Predicted Train {col_idx + 1}', color=color, linestyle='--')
    
    ax2.set_title('Train Data')  # Title for the training data plot
    ax2.legend()  # Add a legend to distinguish actual vs predicted
    ax2.minorticks_on()  # Enable minor ticks for clarity
    
    # Compute y-axis limits for actual vs predicted plot
    all_values = np.concatenate([actual_train.flatten(), predicted_train.flatten()])
    ymin = all_values.min()
    ymax = all_values.max()
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
    ax2.set_ylim(ymin, ymax)
    
    # Enable grid lines for the actual vs predicted plot
    ax2.grid(True, which='both', axis='both')
    
    # Set an overarching title for the figure
    fig.suptitle('Live Loss and Curve Fitting Plot')
    
    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()

#############################################################################################################################

def train_loss_regression(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    actual_train: np.ndarray, 
    predicted_train: np.ndarray,
    regression_column: int = 0,
    figure_size: tuple = (8, 6)
) -> None:
    """
    Plots training loss and regression plot for actual vs predicted values for the training dataset.

    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch during training.
    - loss_train (np.ndarray): Training loss values per epoch.
    - actual_train (np.ndarray): Actual values for the training dataset.
    - predicted_train (np.ndarray): Predicted values for the training dataset.
    - regression_column (int): Column index to use for the regression plot (default is 0).
    - figure_size (tuple): Size of the figure, default is (8, 6).

    Returns:
    - None: Displays the loss and regression plots.
    """
    if not (0 <= regression_column < actual_train.shape[1]):
        raise ValueError(f"Invalid `regression_column`: Must be between 0 and {actual_train.shape[1] - 1}.")
    
    # Create a 2-row subplot grid
    fig, axes = plt.subplots(2, 1, figsize=figure_size)
    
    # Extract individual axes for easy reference
    ax1, ax2 = axes
    
    # Plot training loss with a logarithmic y-scale
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')  # Label the x-axis
    ax1.set_ylabel('Loss')  # Label the y-axis
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')  # Title showing the last training loss
    ax1.yaxis.grid(True, which='minor')  # Enable minor grid for clarity
    ax1.xaxis.grid(False)  # Disable x-axis grid
    
    # Set y-axis limits based on minimum and maximum values of loss
    ymin = float(min(loss_train)) * 0.9  # Set lower limit as 90% of the minimum loss
    ymax = float(max(loss_train)) * 1.1  # Set upper limit as 110% of the maximum loss
    ax1.set_ylim(ymin, ymax)  # Apply limits to the training loss plot

    # Plot the scatter plot for the training set (actual vs predicted values)
    ax2.scatter(actual_train[:, regression_column], predicted_train[:, regression_column],
                color='blue', facecolors='none', label='Train Data')  # Training data points
    
    # Compute min and max values across the actual values to set the plotting range
    min_value = float(min(actual_train[:, regression_column]))
    max_value = float(max(actual_train[:, regression_column]))
    
    # Perform linear regression on the training set
    train_model = LinearRegression()
    train_model.fit(np.array(actual_train[:, regression_column]).reshape(-1, 1), predicted_train[:, regression_column])
    a_train = train_model.coef_[0]  # Slope of the regression line
    b_train = train_model.intercept_  # Intercept of the regression line
    
    # Plot reference line (perfect fit line) and fitted regression line for the training set
    ax2.plot([min_value, max_value], [min_value, max_value], 'r-', label='Reference Line')  # Perfect fit line
    ax2.plot([min_value, max_value], [a_train * min_value + b_train, a_train * max_value + b_train], 'k--', label='Fit Line')  # Regression line
    
    # Set labels and titles for the training scatter plot
    ax2.set_xlabel('Expected Values')
    ax2.set_ylabel('Predicted Values')
    a_train = float(a_train)
    ax2.set_title(f'Train Data: Reg. Coeff = {a_train:.2f}')  # Title showing regression coefficient for training set
    ax2.legend()  # Show legend
    ax2.text(0.5, 0.05, f"Showing regression for column {regression_column + 1}",
             transform=ax2.transAxes, fontsize=10, color='red', ha='center')
    
    # Add grid lines and minor ticks for the scatter plot
    ax2.minorticks_on()
    ax2.grid(True, which='both')
    
    # Set an overarching title for the entire figure
    fig.suptitle('Live Loss and Regression Plot')
    
    # Adjust layout to prevent overlap between titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Display the plots
    plt.show()

#############################################################################################################################

def train_loss_regression_fitting(
    number_of_epochs: int, 
    current_epoch: int, 
    loss_train: np.ndarray, 
    actual_train: np.ndarray, 
    predicted_train: np.ndarray, 
    regression_column: int = 0, 
    figure_size: tuple = (8, 12)
) -> None:
    """
    Plots training loss, actual vs predicted values, and regression fit for a specified column of the training dataset.

    Parameters:
    - number_of_epochs (int): Total number of epochs for the training process.
    - current_epoch (int): The current epoch of training.
    - loss_train (np.ndarray): Array of training loss values.
    - actual_train (np.ndarray): Actual target values for the training set.
    - predicted_train (np.ndarray): Predicted target values for the training set.
    - regression_column (int): Column index to use for the regression plot (default is 0).
    - figure_size (tuple): Figure size, default is (8, 12).

    Returns:
    - None: Displays the plots.
    """
    # Ensure `actual_train` and `predicted_train` are at least 2D
    actual_train = np.atleast_2d(actual_train)
    predicted_train = np.atleast_2d(predicted_train)

    # Check shapes and validity of `regression_column`
    if actual_train.shape != predicted_train.shape:
        raise ValueError("Shape mismatch: `actual_train` and `predicted_train` must have the same shape.")
    if not (0 <= regression_column < actual_train.shape[1]):
        raise ValueError(f"Invalid `regression_column`: Must be between 0 and {actual_train.shape[1] - 1}.")

    num_columns = actual_train.shape[1]
    colors = plt.cm.get_cmap('tab10', num_columns)  # Use distinct colors for columns

    # Create a 3-row grid of subplots
    fig, axes = plt.subplots(3, 1, figsize=figure_size)

    # First row: Training loss vs. epochs (log scale)
    ax1 = axes[0]
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')
    ax1.yaxis.grid(True, which='minor')
    ax1.xaxis.grid(False)
    ax1.set_ylim(min(loss_train) * 0.9, max(loss_train) * 1.1)

    # Second row: Actual vs predicted values for all columns
    ax2 = axes[1]
    for col_idx in range(num_columns):
        color = colors(col_idx)
        ax2.plot(actual_train[:, col_idx], label=f'Actual (Column {col_idx + 1})', color=color, linestyle='-')
        ax2.plot(predicted_train[:, col_idx], label=f'Predicted (Column {col_idx + 1})', color=color, linestyle='--')
    ax2.set_title('Train Data: Actual vs Predicted')
    ax2.legend()
    ax2.grid(True, which='both')
    ax2.minorticks_on()

    # Set y-axis limits for actual vs predicted values
    ymin = min(actual_train.min(), predicted_train.min()) * 0.9
    ymax = max(actual_train.max(), predicted_train.max()) * 1.1
    ax2.set_ylim(ymin, ymax)

    # Third row: Regression fit for the specified column
    ax3 = axes[2]
    actual_col = actual_train[:, regression_column]
    predicted_col = predicted_train[:, regression_column]
    ax3.scatter(actual_col, predicted_col, color='blue', facecolors='none', label='Train Data')

    # Perform linear regression
    min_val, max_val = actual_col.min(), actual_col.max()
    reg_model = LinearRegression()
    reg_model.fit(actual_col.reshape(-1, 1), predicted_col)
    slope, intercept = reg_model.coef_[0], reg_model.intercept_

    # Plot reference and fit lines
    ax3.plot([min_val, max_val], [min_val, max_val], 'r-', label='Reference Line')
    ax3.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], 'k--', label='Fit Line')
    ax3.set_xlabel('Expected Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title(f'Regression Plot: Column {regression_column + 1}\nReg. Coeff = {slope:.2f}')
    ax3.legend()
    ax3.grid(True, which='both')

    # Add notice for the selected column
    ax3.text(0.5, 0.05, f"Showing regression for column {regression_column + 1}",
             transform=ax3.transAxes, fontsize=10, color='red', ha='center')

    # Set overarching title and adjust layout
    fig.suptitle('Training Loss, Curve Fitting, and Regression Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)  # Add more vertical space between subplots

    # Show the figure
    plt.show()

#############################################################################################################################

def train_loss_confusion(
    number_of_epochs: int,
    current_epoch: int,
    Loss_train: np.ndarray,
    train_targets: np.ndarray,
    train_predictions: np.ndarray,
    classes: list,
    figure_size: tuple = (8, 12)
) -> None:
    """
    Plots the training loss curve and confusion matrix for the training dataset.

    Parameters:
    - number_of_epochs (int): Total number of training epochs.
    - current_epoch (int): The current epoch number.
    - Loss_train (np.ndarray): Training loss values over the epochs.
    - train_targets (np.ndarray): Actual labels for the training data.
    - train_predictions (np.ndarray): Predicted labels for the training data.
    - classes (list): List of class labels for the confusion matrix.
    - figure_size (tuple): Figure size, default is (8, 12).

    Returns:
    - None: Displays the loss curve and confusion matrix.
    """
    # Handle binary and multi-class classification
    def convert_predictions_for_binary(predictions):
        """ Convert continuous probabilities to binary class labels (0 or 1). """
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        """ Convert one-hot encoded labels to class indices. """
        if labels.ndim == 2:
            return np.argmax(labels, axis=1)
        return labels

    # Determine if binary classification is being used
    is_binary_classification = train_targets.shape[1] == 1

    if is_binary_classification:
        # Convert binary predictions to 0/1 format
        train_predictions = convert_predictions_for_binary(train_predictions)
        train_targets = train_targets.flatten()
    else:
        # Convert one-hot encoded labels to indices for multi-class classification
        train_targets = convert_one_hot(train_targets)
        train_predictions = convert_one_hot(train_predictions)

    # Set up the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=figure_size)

    # Plot training loss
    axs[0].plot(range(1, current_epoch + 1), Loss_train, color='blue')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Train Loss, last epoch: {Loss_train[-1]:.5f}')
    axs[0].yaxis.grid(True, which='minor')
    axs[0].xaxis.grid(False)

    # Set consistent y-limits for the loss plot
    ymin = float(min(Loss_train)) * 0.9
    ymax = float(max(Loss_train)) * 1.1
    axs[0].set_ylim(ymin, ymax)

    # Compute confusion matrix for training data
    train_cm = confusion_matrix(train_targets, train_predictions)

    # Normalize the confusion matrix
    train_cm_normalized = train_cm.astype('float') / (train_cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # Calculate accuracy
    train_accuracy = accuracy_score(train_targets, train_predictions)

    # Plot the confusion matrix
    def plot_cm(ax, cm, cm_normalized, accuracy, title, classes):
        """ Plot the confusion matrix with raw counts and normalized percentages. """
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted Labels', ylabel='True Labels',
               title=f'{title}\nAccuracy: {accuracy * 100:.2f}%')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells with raw counts and percentages
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j] * 100:.2f}%)",
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "white")

    # Plot the training confusion matrix
    plot_cm(axs[1], train_cm, train_cm_normalized, train_accuracy, 'Train Confusion Matrix', classes)

    # Final adjustments to layout
    fig.suptitle('Live Training Loss and Confusion Matrix')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#############################################################################################################################

def train_loss_index(
    y_train_true: np.array, y_train_pred: np.array, class_labels: list,
    number_of_epochs: int, current_epoch: int, loss_train: np.ndarray,
    figure_size: tuple = (10, 10)
) -> None:
    """
    Function to plot real vs predicted labels for training data and training loss over epochs.
    
    Supports binary and multiclass classification.
    """
    def convert_predictions_for_binary(predictions):
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        if labels.ndim == 2:
            return np.argmax(labels, axis=1)
        return labels

    is_binary_classification = y_train_true.shape[1] == 1

    if is_binary_classification:
        y_train_pred = convert_predictions_for_binary(y_train_pred)
        y_train_true = y_train_true.flatten()
    else:
        y_train_true = convert_one_hot(y_train_true)
        y_train_pred = convert_one_hot(y_train_pred)

    fig, axes = plt.subplots(2, 1, figsize=figure_size)

    # Plot training loss
    axes[0].plot(range(1, current_epoch + 1), loss_train, color='blue')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')
    axes[0].yaxis.grid(True, which='minor')
    axes[0].xaxis.grid(False)

    ymin = float(min(loss_train)) * 0.9
    ymax = float(max(loss_train)) * 1.1
    axes[0].set_ylim(ymin, ymax)

    # Plot real vs predicted labels for training data
    num_train_samples = len(y_train_true)
    num_classes = len(class_labels)

    x_train = np.arange(num_train_samples)
    y_ticks = np.arange(num_classes)

    axes[1].scatter(x_train, y_train_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    axes[1].scatter(x_train, y_train_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Class Label")
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(class_labels)
    axes[1].set_title("Training Data")
    axes[1].legend()

    fig.suptitle("Live Training Loss and Index Plot")

    # Adjust spacing to avoid overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)  # Add more vertical space between subplots
    plt.show()

#############################################################################################################################

def train_loss_index_confusion(
    y_train_true: np.array, y_train_pred: np.array, class_labels: list,
    number_of_epochs: int, current_epoch: int, loss_train: np.ndarray,
    figure_size: tuple = (12, 6)
) -> None:
    """
    Combines loss curve, index plot for real vs. predicted labels, and confusion matrix for training data.
    """
    def convert_predictions_for_binary(predictions):
        return (predictions >= 0.5).astype(int)

    def convert_one_hot(labels):
        return np.argmax(labels, axis=1) if labels.ndim == 2 else labels

    is_binary_classification = y_train_true.shape[1] == 1

    if is_binary_classification:
        y_train_pred = convert_predictions_for_binary(y_train_pred).flatten()
        y_train_true = y_train_true.flatten()
    else:
        y_train_true = convert_one_hot(y_train_true)
        y_train_pred = convert_one_hot(y_train_pred)

    fig = plt.figure(figsize=figure_size)
    fig.suptitle("Training Loss, Real vs Predicted Labels, and Confusion Matrix", fontsize=16, y=1.02)

    # Subplot 1: Training loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(range(1, current_epoch + 1), loss_train, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel(f'Epochs ({current_epoch}/{number_of_epochs})')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Train Loss, last epoch: {loss_train[-1]:.5f}')
    ax1.yaxis.grid(True, which='minor')
    ax1.xaxis.grid(False)
    ymin = float(min(loss_train)) * 0.9
    ymax = float(max(loss_train)) * 1.1
    ax1.set_ylim(ymin, ymax)

    # Subplot 2: Real vs Predicted labels
    ax2 = plt.subplot(2, 2, 3)
    x_train = np.arange(len(y_train_true))
    y_ticks = np.arange(len(class_labels))
    ax2.scatter(x_train, y_train_true, color='blue', marker='o', label="Real Labels", alpha=0.6)
    ax2.scatter(x_train, y_train_pred, color='red', marker='x', label="Predicted Labels", alpha=0.6)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Class Label")
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(class_labels)
    ax2.set_title("Training Data")
    ax2.legend()

    # Subplot 3: Confusion matrix
    ax3 = plt.subplot(1, 2, 2)
    train_cm = confusion_matrix(y_train_true, y_train_pred)
    train_accuracy = accuracy_score(y_train_true, y_train_pred)

    cm_normalized = train_cm.astype('float') / (train_cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    im = ax3.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax3)
    ax3.set(
        xticks=np.arange(len(class_labels)), yticks=np.arange(len(class_labels)),
        xticklabels=class_labels, yticklabels=class_labels,
        xlabel='Predicted', ylabel='True',
        title=f'Train Confusion Matrix\nAccuracy: {train_accuracy * 100:.2f}%'
    )
    for i in range(train_cm.shape[0]):
        for j in range(train_cm.shape[1]):
            ax3.text(j, i, f"{train_cm[i, j]}\n({cm_normalized[i, j] * 100:.1f}%)",
                     ha="center", va="center",
                     color="white" if train_cm[i, j] > train_cm.max() / 2 else "black")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.show()
