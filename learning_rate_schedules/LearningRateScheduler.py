import numpy as np


class ExponentialDecayScheduler:
    """
    Exponential Decay Learning Rate Scheduler.
    Adjusts learning rate based on exponential decay.
    """

    def __init__(self, lr_initial: float = 1e-3, decay_rate: float = 0.1):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float, optional
            Initial learning rate (default is 1e-3).
        decay_rate : float, optional
            Decay rate for exponential adjustment (default is 0.1).
        """
        self.lr_initial = lr_initial
        self.decay_rate = decay_rate

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Adjust the learning rate based on the current epoch.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        return self.lr_initial * (self.decay_rate ** current_epoch)

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class StepDecayScheduler:
    """
    Step Decay Learning Rate Scheduler.
    Reduces the learning rate by a factor after every fixed step of epochs.
    """

    def __init__(self, lr_initial: float = 1e-3, decay_rate: float = 0.5, step_size: int = 5):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float, optional
            Initial learning rate (default is 1e-3).
        decay_rate : float, optional
            Factor by which to reduce the learning rate (default is 0.5).
        step_size : int, optional
            Number of epochs after which the learning rate is reduced (default is 10).
        """
        self.lr_initial = lr_initial
        self.decay_rate = decay_rate
        self.step_size = step_size

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Adjust the learning rate based on the current epoch.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        steps = current_epoch // self.step_size
        return self.lr_initial * (self.decay_rate ** steps)

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class LinearDecayScheduler:
    """
    Linear Decay Learning Rate Scheduler.
    Reduces the learning rate linearly over epochs.
    """

    def __init__(self, lr_initial: float = 1e-3, total_epochs: int = 15):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float, optional
            Initial learning rate (default is 1e-3).
        total_epochs : int, optional
            Total number of epochs for linear decay (default is 100).
        """
        self.lr_initial = lr_initial
        self.total_epochs = total_epochs

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Adjust the learning rate based on the current epoch.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        return max(self.lr_initial * (1 - current_epoch / self.total_epochs), 0.0)

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class PolynomialDecayScheduler:
    """
    Polynomial Decay Learning Rate Scheduler.
    Reduces the learning rate based on a polynomial function.
    """

    def __init__(self, lr_initial: float = 1e-3, total_epochs: int = 15, power: float = 2.0):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float, optional
            Initial learning rate (default is 1e-3).
        total_epochs : int, optional
            Total number of epochs for polynomial decay (default is 100).
        power : float, optional
            The power of the polynomial (default is 2.0).
        """
        self.lr_initial = lr_initial
        self.total_epochs = total_epochs
        self.power = power

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Adjust the learning rate based on the current epoch.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        return max(self.lr_initial * ((1 - current_epoch / self.total_epochs) ** self.power), 0.0)

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class CosineAnnealingScheduler:
    """
    Cosine Annealing Learning Rate Scheduler.
    Adjusts the learning rate following a cosine annealing schedule.
    """

    def __init__(self, lr_initial: float = 1e-3, total_epochs: int = 100, lr_min: float = 1e-6):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float, optional
            Initial learning rate (default is 1e-3).
        total_epochs : int, optional
            Total number of epochs for the training (default is 100).
        lr_min : float, optional
            Minimum learning rate value (default is 1e-6).
        """
        self.lr_initial = lr_initial
        self.total_epochs = total_epochs
        self.lr_min = lr_min

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Adjust the learning rate based on the current epoch using cosine annealing.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        # Compute the cosine annealed learning rate
        cosine_decay = 0.5 * (1 + np.cos(np.pi * current_epoch / self.total_epochs))
        lr = self.lr_min + (self.lr_initial - self.lr_min) * cosine_decay
        return lr

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class CyclicCosineAnnealingScheduler:
    """
    Cyclic Cosine Annealing Learning Rate Scheduler with Restarts.
    """

    def __init__(self, lr_initial: float, total_epochs: int, lr_min: float, cycles: int):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float
            Initial learning rate.
        total_epochs : int
            Total number of epochs for training.
        lr_min : float
            Minimum learning rate.
        cycles : int
            Number of cosine annealing cycles within the total epochs.
        """
        self.lr_initial = lr_initial
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.cycles = cycles
        self.epochs_per_cycle = total_epochs // cycles

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Calculate the learning rate for the given epoch.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        # Determine the current cycle
        cycle_epoch = current_epoch % self.epochs_per_cycle

        # Compute cosine annealed learning rate
        cos_inner = np.pi * cycle_epoch / self.epochs_per_cycle
        return self.lr_min + 0.5 * (self.lr_initial - self.lr_min) * (1 + np.cos(cos_inner))

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class CosineAnnealingWithWarmUpScheduler:
    """
    Cosine Annealing Learning Rate Scheduler with Warm-Up Phase.
    """

    def __init__(self, lr_initial: float, lr_max: float, warmup_epochs: int, total_epochs: int, lr_min: float):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float
            Initial learning rate for the warm-up phase.
        lr_max : float
            Maximum learning rate (achieved at the end of the warm-up phase).
        warmup_epochs : int
            Number of epochs for the warm-up phase.
        total_epochs : int
            Total number of epochs for training (including warm-up).
        lr_min : float
            Minimum learning rate for cosine annealing.
        """
        self.lr_initial = lr_initial
        self.lr_max = lr_max
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.annealing_epochs = total_epochs - warmup_epochs

    def __call__(self, current_epoch: int, current_loss: float = None, **kwargs) -> float:
        """
        Calculate the learning rate for the given epoch.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float, optional
            The current loss value (not used in this method).

        Returns:
        -------
        float
            The updated learning rate.
        """
        if current_epoch < self.warmup_epochs:
            # Warm-up phase: linear increase
            lr = self.lr_initial + (self.lr_max - self.lr_initial) * (current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            annealing_epoch = current_epoch - self.warmup_epochs
            cos_inner = np.pi * annealing_epoch / self.annealing_epochs
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(cos_inner))
        
        return lr

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class ReduceOnPlateauScheduler:
    """
    Reduce on Plateau Learning Rate Scheduler.
    Adjusts the learning rate when the loss plateaus for a specified number of epochs.
    """

    def __init__(self, lr_initial: float = 1e-3, factor: float = 0.5, patience: int = 5, min_lr: float = 1e-6):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        lr_initial : float, optional
            Initial learning rate (default is 1e-3).
        factor : float, optional
            Factor by which to reduce the learning rate (default is 0.5).
        patience : int, optional
            Number of epochs to wait for loss improvement before reducing LR (default is 5).
        min_lr : float, optional
            Minimum learning rate (default is 1e-6).
        """
        self.lr = lr_initial
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = np.inf

    def __call__(self, current_epoch: int, current_loss: float, **kwargs) -> float:
        """
        Adjust the learning rate based on the current loss.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float
            The current loss value.

        Returns:
        -------
        float
            The updated learning rate.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self.wait = 0
        return self.lr

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class CombinedScheduler:
    """
    Combined Time-Based and Reduce on Plateau Scheduler.
    Combines a time-based scheduler with a loss-based scheduler.
    """

    def __init__(self, time_scheduler, loss_scheduler):
        """
        Initialize the scheduler.

        Parameters:
        ----------
        time_scheduler : object
            A time-based scheduler object.
        loss_scheduler : object
            A loss-based scheduler object (e.g., ReduceOnPlateauScheduler).
        """
        self.time_scheduler = time_scheduler
        self.loss_scheduler = loss_scheduler

    def __call__(self, current_epoch: int, current_loss: float, **kwargs) -> float:
        """
        Adjust the learning rate using both time-based and loss-based strategies.

        Parameters:
        ----------
        current_epoch : int
            The current epoch in training.
        current_loss : float
            The current loss value.

        Returns:
        -------
        float
            The updated learning rate.
        """
        lr_time = self.time_scheduler(current_epoch, current_loss, **kwargs)
        lr_loss = self.loss_scheduler(current_epoch, current_loss, **kwargs)
        return min(lr_time, lr_loss)
