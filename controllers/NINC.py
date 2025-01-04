import numpy as np
import inspect
from IPython.display import clear_output
from utils import plot_open_loop, index_n_close_loop
from model import Online_NN_section
import scipy

class NINC_plant:
    """
    A class representing a nonlinear plant model with state and output equations.
    
    Attributes:
    - state_equation: Function defining the state dynamics.
    - output_equation: Function defining the system's output behavior.
    - input_size: Integer, number of input variables.
    - output_size: Integer, number of output variables.
    - state_size: Integer, number of state variables.
    - state: NumPy array representing the state vector.
    - output: NumPy array representing the output vector.
    """
    
    def __init__(self, state_equation: callable, output_equation: callable,
                 input_size: int, output_size: int, state_size: int):
        """
        Initializes the NINC_plant object and validates the input functions.

        :param state_equation: Function defining the system state dynamics.
        :param output_equation: Function defining the system output.
        :param input_size: Number of input variables.
        :param output_size: Number of output variables.
        :param state_size: Number of state variables.
        """
        
        print('checking state_equation argument...')
        if not inspect.isfunction(state_equation):  # Ensure the state equation is a function
            raise TypeError("state_equation must be a Python function defined using def.")
        print('checking state_equation done.')
        
        print('checking output_equation argument...')
        if not inspect.isfunction(output_equation):  # Ensure the output equation is a function
            raise TypeError("output_equation must be a Python function defined using def.")
        print('checking output_equation done.')
        
        # Store provided functions and parameters
        self.state_equation = state_equation
        self.output_equation = output_equation
        self.input_size = input_size
        self.output_size = output_size
        self.state_size = state_size
        
        # initialize perturbation for numerical jaccobian calculation
        self.perturb = 1e-1
        
        # Initialize state and output vectors with zeros
        self.state = np.zeros((state_size, 1))  # State vector
        self.output = np.zeros((output_size, 1))  # Output vector

    #################################################################

    def jacob(self, t: float, dt: float, state: np.ndarray, input: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian of the output with respect to the input (dy/du) in a nonlinear state-space system 
        using numerical differentiation.

        This function computes:
        1. `dodx`: Sensitivity of the output with respect to the state (dy/dx).
        2. `dxdu`: Sensitivity of the state with respect to the input (dx/du).
        3. `dodu`: Direct sensitivity of the output with respect to the input (dy/du).
        4. Computes the final Jacobian using: dodu + dodx @ dxdu.

        :param t: Current time step.
        :param dt: Discretization time step.
        :param state: Current state vector (shape: (state_size,)).
        :param input: Current input vector (shape: (input_size,)).
        :return: Jacobian matrix (output_size x input_size).
        """

        s = state.ravel()  # Ensure state is a 1D array
        u = input.ravel()  # Ensure input is a 1D array

        # Initialize matrices for sensitivities
        dodx = np.zeros((self.output_size, self.state_size))  # dy/dx (output sensitivity to state)
        dodu = np.zeros((self.output_size, self.input_size))  # dy/du (direct output sensitivity to input)
        dxdu = np.zeros((self.state_size, self.input_size))  # dx/du (state sensitivity to input)

        # Compute dy/dx using finite differences
        for i in range(self.state_size):
            s_plus = s.copy()
            s_minus = s.copy()
            s_plus[i] += self.perturb  # Perturb the i-th state positively
            s_minus[i] -= self.perturb  # Perturb the i-th state negatively

            o_plus = self.output_equation(t, s_plus, u)  # Output with perturbed state (positive)
            o_minus = self.output_equation(t, s_minus, u)  # Output with perturbed state (negative)

            dodx[:, i] = 0.5 * (o_plus - o_minus) / self.perturb  # Central difference approximation for dy/dx

        # Compute dx/du using finite differences
        for i in range(self.input_size):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[i] += self.perturb  # Perturb the i-th input positively
            u_minus[i] -= self.perturb  # Perturb the i-th input negatively

            s_plus = dt * self.state_equation(t, s, u_plus)  # Compute state change with perturbed input (positive)
            s_minus = dt * self.state_equation(t, s, u_minus)  # Compute state change with perturbed input (negative)

            dxdu[:, i] = 0.5 * (s_plus - s_minus) / self.perturb  # Central difference approximation for dx/du

        # Compute dy/du using finite differences
        for i in range(self.input_size):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[i] += self.perturb  # Perturb the i-th input positively
            u_minus[i] -= self.perturb  # Perturb the i-th input negatively

            o_plus = self.output_equation(t, s, u_plus)  # Output with perturbed input (positive)
            o_minus = self.output_equation(t, s, u_minus)  # Output with perturbed input (negative)

            dodu[:, i] = 0.5 * (o_plus - o_minus) / self.perturb  # Central difference approximation for dy/du

        # Compute the final Jacobian using the chain rule: dy/du = dodu + dodx @ dxdu
        return dodu + dodx @ dxdu

    #################################################################

    def reset_memory(self) -> None:
        """
        Resets the internal state memory of the plant by setting all state values to zero.

        This ensures that the plant starts from a clean slate for new simulations or control iterations.
        """
        self.state *= 0  # Reset the state vector to all zeros.
        self.output *= 0 # Reset output vector to zero

    #################################################################

    def __call__(self, t: float, dt: float, input: np.ndarray):
        """
        Calls the plant model to compute the next state and output based on the provided inputs.

        :param t: Current time step as a float.
        :param dt: Time step duration as a float.
        :param input: Input vector as a NumPy array.

        :return: Tuple containing:
                - output: Updated output vector as a NumPy array.
                - state: Updated state vector as a NumPy array.

        :raises ValueError: If input size does not match the expected input_size.
        :raises ValueError: If `t` is not a scalar (i.e., not a single float value).
        """

        # Validate that the input size matches the expected input_size attribute
        if input.size != self.input_size:
            raise ValueError(
                f"Input vector length ({input.size}) does not match the expected input_size attribute ({self.input_size}). "
                "Ensure the input dimension is correct."
            )

        # Validate that t is a single scalar value (not an array)
        if np.size(t) != 1:
            raise ValueError(
                "The argument `t` is expected to be a single float value, but an array was provided. "
                "Ensure that `t` is a scalar."
            )

        # Convert input to a 1D array to ensure consistency
        u = input.ravel()

        # Convert the current state to a 1D array for processing
        s = self.state.ravel()

        # Compute the first Runge-Kutta term (k1)
        # This evaluates the state equation at the current state and input
        k1 = self.state_equation(t, s, u)

        # Compute the second Runge-Kutta term (k2)
        # Evaluates the state equation at the midpoint of the step using k1
        k2 = self.state_equation(t + dt / 2, s + dt * k1 / 2, u)

        # Compute the third Runge-Kutta term (k3)
        # Again evaluates the state equation at the midpoint, but now using k2
        k3 = self.state_equation(t + dt / 2, s + dt * k2 / 2, u)

        # Compute the fourth Runge-Kutta term (k4)
        # Evaluates the state equation at the end of the step using k3
        k4 = self.state_equation(t + dt, s + dt * k3, u)

        # Compute the next state using the weighted sum of the RK4 terms
        # This is the classic 4th-order Runge-Kutta integration formula:
        # s_new = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        new_state = self.state.ravel() + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Store the updated state, reshaping it back to column vector form
        self.state = new_state.reshape((-1, 1))

        # Compute the output using the provided output equation
        self.output = self.output_equation(t, self.state.ravel(), u).reshape((-1, 1))

        return self.output, self.state
    
    #################################################################

    def simulate(self, input_vectors: np.ndarray, time_vector: np.ndarray, initial_states: np.ndarray = None, **kwargs):
        """
        Simulates the open-loop response of the system over a given time vector with specified input signals.

        :param input_vectors: A NumPy array of shape (input_size, num_time_samples) containing the input signals.
        :param time_vector: A 1D NumPy array containing time steps for simulation.
        :param initial_state: A numpy array as system initial states
        :param **kwargs: whether to plot or not and figure size of the plot

        :return: A dictionary containing:
                - 'output': The system output matrix of shape (output_size, num_time_samples).
                - 'state': The system state matrix of shape (state_size, num_time_samples).

        :raises ValueError: If `time_vector` is not a 1D array.
        :raises ValueError: If `input_vectors` does not match the expected input size.
        :raises ValueError: If `input_vectors` does not match the number of time samples.
        """

        # Ensure that the time_vector is one-dimensional
        if np.ndim(time_vector) != 1:
            raise ValueError(
                f"Expected time_vector to have only one dimension, but got {np.ndim(time_vector)} dimensions. "
                "Ensure that time_vector is a 1D array."
            )

        # Validate that input_vectors has the correct number of rows (matching input_size)
        if input_vectors.shape[0] != self.input_size:
            raise ValueError(
                f"The number of rows in the input matrix ({input_vectors.shape[0]}) "
                f"must match the system's expected input size ({self.input_size})."
            )

        # Validate that input_vectors has the correct number of columns (matching time_vector size)
        if input_vectors.shape[1] != time_vector.size:
            raise ValueError(
                f"The number of columns in the input matrix ({input_vectors.shape[1]}) "
                f"must match the number of time samples ({time_vector.size})."
            )

        # Initial states
        self.state = initial_states if initial_states is not None else self.state

        # Flatten time_vector to ensure it is a 1D array
        time_vector = time_vector.ravel()

        # Initialize output matrix with zeros (output_size x number of time samples)
        output = np.zeros((self.output_size, time_vector.size))

        # Initialize state matrix with zeros (state_size x number of time samples)
        state = np.zeros((self.state_size, time_vector.size))
        state[:, [0]] = self.state  # Set initial state values
        # print(self.state)

        # Iterate over time steps, starting from the second time step
        for ind_t, t in enumerate(time_vector[1:]):
            t = float(t)  # Ensure time is a float
            u = input_vectors[:, [ind_t + 1]]  # Extract current input vector
            dt = time_vector[ind_t + 1] - time_vector[ind_t]  # Compute time step size

            # Compute system response (new state and output) using the __call__ method
            self(t, dt, u)
            output[:, [ind_t + 1]], state[:, [ind_t + 1]] = self.output, self.state

        # Plot the open-loop response
        plot = kwargs.get('plot', True)
        if plot:
            plot_open_loop(state, output, input_vectors, time_vector, continuous=True, fig_size=kwargs.get('fig_size', (8, 6)))

        # Return the computed state and output trajectories
        return {'output': output, 'state': state}

#############################################################################################################################

class NINC_close_loop:
    """
    Close-Loop Direct-Inverse Neuro-Controller (NINC) structure.

    This class implements a closed-loop control system where a neural network controller 
    interacts with an open-loop plant model (`NINC_plant`). The controller takes delayed 
    system outputs as input and generates control signals to be applied to the plant.

    Attributes:
    -----------
    G : NINC_plant
        The open-loop plant model that is controlled.
    C : Compile
        The neural network-based controller, which must be an instance of the `Compile` class.
    Error : np.ndarray
        Stores the error history for delayed feedback, with rows representing system outputs 
        and columns representing delays.
    """

    def __init__(self, controller, open_loop_system, n_delay: int):
        """
        Initializes the closed-loop system by validating and linking the controller with the open-loop plant.

        Parameters:
        -----------
        controller : Compile
            The neural network-based controller, expected to be an instance of the `Compile` class.
        open_loop_system : NINC_plant
            The open-loop plant model, expected to be an instance of `NINC_plant`.
        n_delay : int
            The number of delay steps for incorporating past errors into the controller's input.

        Raises:
        -------
        ValueError
            If `open_loop_system` is not an instance of `NINC_plant`.
        ValueError
            If the controller's output size does not match the open-loop system's input size.
        ValueError
            If the controller's input size does not match the expected size based on system outputs and delays.
        """

        # Validate that the provided open-loop system is an instance of NINC_plant
        if not isinstance(open_loop_system, NINC_plant):
            raise ValueError(
                "The open-loop plant system must be an instance of the `NINC_plant` class, "
                "which should be defined in the same directory as `NINC_close_loop`."
            )

        # Store the open-loop system and the controller
        self.G = open_loop_system  # Plant (open-loop system)
        self.C = controller  # Neural network-based controller (Compile object)

        # Ensure that the controller's output size matches the plant's input size
        if controller.model[-1].output_size != open_loop_system.input_size:
            raise ValueError(
                f"Controller output size ({controller.model[-1].output_size}) must match "
                f"the open-loop system input size ({open_loop_system.input_size})."
            )

        # Ensure that the controller's input size accounts for the delayed system outputs
        expected_input_size = open_loop_system.output_size * (n_delay + 1)
        if controller.model[0].input_size != expected_input_size:
            raise ValueError(
                f"Controller input size ({controller.model[0].input_size}) must be equal to "
                f"the open-loop system output size ({open_loop_system.output_size}) multiplied by "
                f"(n_delay + 1) ({n_delay + 1})."
            )

        # Initialize the error matrix to store past errors for delayed feedback
        # Rows represent system outputs, columns represent delay steps
        self.Error = np.zeros((open_loop_system.output_size, n_delay + 1))

        # Confirmation message indicating successful initialization
        print(
            "Closed-loop system implementation was successful.\n"
        )

    #################################################################

    def reset_memory(self) -> None:
        """
        Resets the memory states of both the controller and the open-loop system.

        This is useful for re-initializing the system when starting a new control experiment
        or simulation.
        """
        self.C.reset_memory()  # Reset controller memory
        self.G.reset_memory()  # Reset plant memory
        self.Error *= 0  # Reset Controller input
        self.C.model[-1].output *= 0  # Reset controll effort
             
    #################################################################

    def __call__(self, t: float, dt: float, references: np.ndarray):
        """
        Executes a closed-loop control step, computing the control effort and updating the system state.

        :param t: Current time step as a float.
        :param dt: Time step duration as a float.
        :param references: Reference signal matrix as a NumPy array, expected to match the open-loop system output size.

        :return: A dictionary containing:
                - 'output': The updated output matrix of the system.
                - 'state': The updated state matrix of the system.
                - 'control effort': The computed control effort applied to the system.

        :raises ValueError: If `t` is not a single scalar value.
        :raises ValueError: If `references` does not match the open-loop system output size.
        """

        # Ensure t is a scalar (single float value)
        if np.size(t) != 1:
            raise ValueError(
                f"Expected `t` to be a single float value, but received a vector of size {np.ndim(t)}. "
                "Ensure `t` is a scalar."
            )

        # Validate that the number of reference elements matches the expected output size of the open-loop system
        if references.size != self.G.output_size:
            raise ValueError(
                f"The number of elements in the reference matrix ({references.size}) "
                f"must match the output size of the open-loop system ({self.G.output_size})."
            )

        # Reshape references into a column vector
        references = references.reshape((-1, 1))

        # Compute the error signal (difference between reference and actual output)
        e_k = references - self.G.output

        # Update the error history matrix, shifting previous errors and adding the new error
        self.Error = np.concatenate((e_k, self.Error[:, :-1]), axis=1)

        # Compute the control effort using the controller
        # The controller takes the transposed error history reshaped into a single-row vector
        U = self.C(self.Error.T.reshape((1, -1))).reshape((-1, 1))

        # Apply the control effort to the system and obtain the new output and state
        output, state = self.G(t, dt, U)

        # Return the updated system values
        return {'output': output, 'state': state, 'control effort': U}
    
    #################################################################

    def simulate(self, reference_vectors: np.ndarray, time_vector: np.ndarray, plot: bool = False, **kwargs):
        """
        Simulates the closed-loop system response over a given time vector with specified reference signals.

        :param reference_vectors: A NumPy array of shape (output_size, num_time_samples) containing the reference signals.
        :param time_vector: A 1D NumPy array containing time steps for simulation.
        :param plot: Boolean flag indicating whether to generate plots of the system's response.
        :param kwargs: Optional keyword arguments for plot customization, including:
                    - 'fig_size1': Figure size for state and output response plot.
                    - 'fig_size2': Figure size for control effort plot.

        :return: A dictionary containing:
                - 'output': The system output matrix of shape (output_size, num_time_samples).
                - 'state': The system state matrix of shape (state_size, num_time_samples).
                - 'control effort': The control input matrix of shape (input_size, num_time_samples).
                - 'X_controller': The controller's input matrix used for training or analysis.

        :raises ValueError: If `time_vector` is not a 1D array.
        :raises ValueError: If `reference_vectors` does not match the expected output size.
        :raises ValueError: If `reference_vectors` does not match the number of time samples.
        """

        # Ensure that the time_vector is one-dimensional
        if np.ndim(time_vector) != 1:
            raise ValueError(
                f"Expected `time_vector` to be a one-dimensional array, but received {np.ndim(time_vector)} dimensions. "
                "Ensure that `time_vector` is a 1D NumPy array."
            )

        # Validate that reference_vectors has the correct number of rows (matching output_size)
        if reference_vectors.shape[0] != self.G.output_size:
            raise ValueError(
                f"The number of rows in the reference matrix ({reference_vectors.shape[0]}) "
                f"must match the output size of the open-loop system ({self.G.output_size})."
            )

        # Validate that reference_vectors has the correct number of columns (matching time_vector size)
        if reference_vectors.shape[1] != time_vector.size:
            raise ValueError(
                f"The number of columns in the reference matrix ({reference_vectors.shape[1]}) "
                f"must match the number of time samples ({time_vector.size})."
            )

        # Ensure time_vector is a flattened 1D array
        time_vector = time_vector.ravel()

        # Reset controller memory before simulation starts
        self.C.reset_memory()

        # Initialize output matrix (output_size x num_time_samples)
        output = np.zeros((self.G.output_size, time_vector.size))
        output[:, [0]] = self.G.output  # Set initial output values

        # Initialize state matrix (state_size x num_time_samples)
        state = np.zeros((self.G.state_size, time_vector.size))
        state[:, [0]] = self.G.state  # Set initial state values

        # Initialize control effort matrix (input_size x num_time_samples)
        U_t = np.zeros((self.G.input_size, time_vector.size))

        # Initialize controller input history matrix (num_time_samples x controller input size)
        X_controller = np.zeros((time_vector.size, self.C.model[0].input_size))

        self.Error *= 0  # Reset Controller input

        # Iterate over time steps, starting from the second time step
        for ind_t, t in enumerate(time_vector[1:]):
            t = float(t)  # Ensure time is a float

            # Compute error between reference and actual output
            e_k = reference_vectors[:, [ind_t + 1]] - self.G.output

            # Update error history by shifting previous errors and adding the new error
            self.Error = np.concatenate((e_k, self.Error[:, :-1]), axis=1)

            # Store the controller input history
            X_controller[ind_t] = self.Error.T.ravel()

            # Compute the control effort using the neural network controller
            U = self.C(X_controller[ind_t].reshape((1, -1))).reshape((-1, 1))

            # Store control effort in matrix
            U_t[:, [ind_t + 1]] = U.reshape((-1, 1))

            # Compute time step size
            dt = time_vector[ind_t + 1] - time_vector[ind_t]

            # Apply control effort to the system and obtain new output and state
            output[:, [ind_t + 1]], state[:, [ind_t + 1]] = self.G(t, dt, U)

        # If plotting is enabled, generate response and control effort plots
        if plot:
            loss = [np.mean(np.square(output-reference_vectors))]
            index_n_close_loop(
                        loss, None, None,
                        reference_vectors, output,
                        time_vector, state, U_t,
                        continuous=True,
                        fig_size=kwargs.get('fig_size', (8, 6))
                        )

        # Return the computed simulation data
        return {'output': output, 'state': state, 'control effort': U_t, 'X_controller': X_controller}
    
    #################################################################

    def online_train(
        self, 
        reference_vectors: np.ndarray, 
        time_vector: np.ndarray, 
        Loss_function, 
        initial: str = 'zeros',
        epoch: int = 1, 
        method: str = 'Adam', 
        learning_rate = 1e-3, 
        **kwargs):
        """
        Performs online training for a closed-loop neuro-controller system.

        :param reference_vectors: A NumPy array of shape (output_size, num_time_samples) containing the reference signals.
        :param time_vector: A 1D NumPy array containing time steps for training.
        :param Loss_function: A callable function representing the loss function used in training.
        :param initial: Initialization strategy for the plant state memory. Options:
                        - 'zeros': Resets the plant state to zero.
                        - 'random': Initializes the state randomly.
                        - 'preserve': Retains the previous state.
        :param epoch: The number of training epochs to run (default: 1).
        :param method: Optimization method to use (default: 'Adam'). Supported methods include:
                        - 'EKF' (Extended Kalman Filter)
                        - 'Levenberg-Marquardt'
                        - 'Gauss-Newton'
                        - Other standard optimizers.
        :param learning_rate: The learning rate, which can be a fixed float value or a callable function that adapts per epoch.
        :param kwargs: Additional keyword arguments for optimizer settings and plotting options.

        :return: A dictionary containing:
                - 'loss': A list of loss values recorded at different stages of training.

        :raises ValueError: If `time_vector` is not a 1D array.
        :raises ValueError: If `reference_vectors` does not match the expected output size.
        :raises ValueError: If `reference_vectors` does not match the number of time samples.
        :raises ValueError: If the optimization method provided is not supported.
        :raises ValueError: If an unsupported initial state assignment is provided.
        """

        # Ensure that the time_vector is one-dimensional
        if np.ndim(time_vector) != 1:
            raise ValueError(
                f"Expected `time_vector` to be a one-dimensional array, but received {np.ndim(time_vector)} dimensions. "
                "Ensure that `time_vector` is a 1D NumPy array."
            )

        # Validate that reference_vectors has the correct number of rows (matching output_size)
        if reference_vectors.shape[0] != self.G.output_size:
            raise ValueError(
                f"The number of rows in the reference matrix ({reference_vectors.shape[0]}) "
                f"must match the output size of the open-loop system ({self.G.output_size})."
            )

        # Validate that reference_vectors has the correct number of columns (matching time_vector size)
        if reference_vectors.shape[1] != time_vector.size:
            raise ValueError(
                f"The number of columns in the reference matrix ({reference_vectors.shape[1]}) "
                f"must match the number of time samples ({time_vector.size})."
            )

        loss = []  # List to store loss values for each epoch

        # Initialize an online version of the controller neural network
        self.C = Online_NN_section(self.C.model)

        # Gradient cliping
        # max_norm_grad = kwargs.get('max_norm_grad', 1e-6)

        # Determine how frequently to record loss updates based on time_percent parameter
        time_percent = kwargs.get('time_percent', 1.0)
        slice_size = int(time_percent * time_vector.size)

        # Training loop over epochs
        for current_epoch in range(epoch):

            self.Error *= 0  # Reset Controller input

            # Reset memory for loss function if applicable
            if hasattr(Loss_function, 'memory'):
                Loss_function.reset_memory()

            # Reset controller memory before training begins
            self.C.reset_memory()

            # Initialize matrices to store training results
            U = np.zeros((self.G.input_size, time_vector.size))  # Control effort
            output = np.zeros((self.G.output_size, time_vector.size))  # System output
            state = np.zeros((self.G.state_size, time_vector.size))  # System state

            # Store the initial state of the plant
            initial_state = self.G.state

            # Determine the learning rate for the current epoch
            if isinstance(learning_rate, (int, float)):
                lr = learning_rate
            else:
                lr = learning_rate(current_epoch, loss[-1] if loss else 1.0)  # Adaptive learning rate

            # Iterate through time steps
            for i in range(time_vector.size - 1):

                # Extract reference output for the next time step
                data_Y = reference_vectors[:, [i + 1]].copy()

                # Compute time step and time duration
                time = time_vector[i + 1]
                dt = time_vector[i + 1] - time_vector[i]

                # Compute system response (output, state, and control effort)
                out = self(time, dt, data_Y)
                output[:, [i + 1]] = out["output"]
                state[:, [i + 1]] = out["state"]
                U[:, [i + 1]] = out['control effort']

                # Compute error and update gradients
                if hasattr(Loss_function, 'memory'):
                    _ = Loss_function.forward(out['output'].T, data_Y.T)
                    error = Loss_function.backward()
                else:
                    error = Loss_function.backward(out['output'].T, data_Y.T)

                # Calculate derivatives    
                jacob = self.G.jacob(time, dt, out['state'].ravel(), out['control effort'].ravel())
                # print(jacob)
                # norm = np.linalg.norm(jacob)
                # if norm > max_norm_grad:
                #     coe = max_norm_grad / norm
                #     jacob *= coe

                error = error @ jacob

                # Apply optimization method to update controller weights
                if method not in ['EKF', 'Levenberg-Marquardt', 'Gauss-Newton']:
                    self.C.optimizer_init(method=method, **kwargs)
                    self.C.backward(error, learning_rate=lr)
                elif method == 'EKF':
                    self.C.optimizer_init("SGD")
                    self.C.EKF(error, Q=kwargs.get('Q', None), R=kwargs.get('R', None), P=kwargs.get('P', None), learning_rate=lr)
                elif method == 'Levenberg-Marquardt':
                    self.C.optimizer_init("SGD")
                    self.C.levenberg_mar(error, learning_rate=lr, gamma=kwargs.get('gamma', 0.99))
                elif method == 'Gauss-Newton':
                    self.C.optimizer_init("SGD")
                    self.C.gauss_newton(error, learning_rate=lr)
                else:
                    raise ValueError(
                        f"The optimization method `{method}` is not supported. "
                        "Please refer to the optimizers directory for supported methods."
                    )

                # Periodically evaluate and record loss
                if (i + 1) % slice_size == 0 or i == time_vector.size - 2:
                    if hasattr(Loss_function, 'memory'):
                        Loss_function.reset_memory()
                    losss = Loss_function.forward(output[:, :i + 2].T, reference_vectors[:, :i + 2].T, inference=True)
                    loss.append(losss)

                    if np.isnan(losss):
                        raise ValueError('NaN occurence')

                    # Plot training progress
                    ind = max(0, i + 2 - slice_size)
                    index_n_close_loop(
                        loss, current_epoch + 1, epoch,
                        reference_vectors[:, ind:i + 2], output[:, ind:i + 2],
                        time_vector[ind:i + 2], state[:, ind:i + 2], U[:, ind:i + 2],
                        continuous=True,
                        fig_size=kwargs.get('fig_size', (10, 8))
                    )
                    clear_output(wait=True)
                    _ = loss.pop()

            # Reset or preserve the system's initial state based on user preference
            if initial == 'zeros':
                self.G.reset_memory()
            elif initial == 'random':
                self.G.reset_memory()
                self.G.state = np.random.uniform(self.G.state.shape)
            elif initial == 'preserve':
                self.G.reset_memory()
                self.G.state = initial_state
            else:
                raise ValueError(
                    f"The specified initial state assignment `{initial}` is not supported. "
                    "Valid options are 'zeros', 'random', or 'preserve'."
                )

            # Append final loss value
            loss.append(losss)

        return {'loss': loss}

    #################################################################

    def continuous_simulation(
        self, 
        time_vector: np.ndarray, 
        reference_vector: np.ndarray, 
        sampling_time: float, 
        samp_per_dt: int = 3, 
        initial_condition = None,
        **kwargs
    ):
        """
        Simulates a continuous-time MIMO system with a discrete-time controller using ODE45 (Runge-Kutta),
        capturing finer resolution states, outputs, and control efforts.

        :param time_vector: 1D NumPy array representing discrete simulation time steps.
        :param reference_vector: 2D NumPy array representing desired reference trajectory (output_size, num_time_samples).
        :param sampling_time: Controller update sampling time.
        :param samp_per_dt: Optional number of finer time steps per `dt` (must be > 0).
        :param initial_condition: Optional initial state vector. Defaults to zeros.
        :param **kwargs: Plot result and it's options

        :return: Dictionary containing:
                - 'output': Simulated system outputs at discrete time steps.
                - 'state': Simulated system states at discrete time steps.
                - 'control effort': Applied control inputs at discrete time steps.
                - 'output_conti': High-resolution outputs.
                - 'state_conti': High-resolution states.
                - 'control_effort_conti': High-resolution control inputs.
                - 'time_vector_conti': Corresponding high-resolution time vector.
        """
        self.reset_memory()

        # Ensure time_vector is 1D
        time_vector = time_vector.ravel()

        # Extract sizes from self.G (plant model)
        output_size, num_time_samples = self.G.output_size, time_vector.size
        state_size, input_size = self.G.state_size, self.G.input_size

        # Whether if to plot or not
        plot = kwargs.get('plot', True)

        # Compute new high-resolution time vector
        time_vector_conti = np.linspace(time_vector[0], time_vector[-1], num=num_time_samples * samp_per_dt)

        # Initialize high-resolution storage arrays
        state_conti = np.zeros((state_size, num_time_samples * samp_per_dt))
        output_conti = np.zeros((output_size, num_time_samples * samp_per_dt))

        # Initialize discrete storage arrays
        state_matrix = np.zeros((state_size, num_time_samples))
        output_matrix = np.zeros((output_size, num_time_samples))
        control_effort_matrix = np.zeros((input_size, num_time_samples))

        # Set initial condition
        state = np.zeros(state_size) if initial_condition is None else initial_condition

        # Compute controller update interval
        last_update_time = 0  # Ensure controller updates at t > 0 and not t = 0

        self.Error *= 0  # Reset Controller input

        # Simulation loop
        for i in range(num_time_samples - 1):
            t_start, t_end = time_vector[i], time_vector[i + 1]
            t_span = np.linspace(t_start, t_end, samp_per_dt)

            # Check if controller should update
            if int(time_vector[i] / sampling_time) > int(last_update_time / sampling_time):
                reference = reference_vector[:, i].reshape((-1, 1))
                e_k = reference - output_matrix[:, i].reshape((-1, 1))
                self.Error = np.concatenate((e_k, self.Error[:, :-1]), axis=1)
                control_input = self.C(self.Error.T.reshape((1, -1))).reshape((-1, 1))  # Compute new control effort
                last_update_time = time_vector[i]  # Store last update time
            else:
                control_input = control_effort_matrix[:, i - 1]  # Keep previous control effort

            control_effort_matrix[:, [i]] = control_input.reshape((-1, 1))  # Store discrete-time control effort

            # Solve ODE45 (Runge-Kutta method) with higher resolution
            sol = scipy.integrate.solve_ivp(
                lambda t, x: self.G.state_equation(t, x, control_input.ravel()), 
                (t_start, t_end), 
                state.ravel(), 
                t_eval=t_span, 
                method='RK45'
            )

            # Update state to final value from integration
            state = sol.y[:, -1]
            state_matrix[:, i + 1] = state

            # Store high-resolution state trajectory
            state_conti[:, i * samp_per_dt:(i + 1) * samp_per_dt] = sol.y

            # Compute output for both discrete and high-resolution data
            for j, t_j in enumerate(t_span):
                output_conti[:, i * samp_per_dt + j] = self.G.output_equation(t_j, sol.y[:, j], control_input.ravel())

            output_matrix[:, i + 1] = output_conti[:, (i + 1) * samp_per_dt - 1]  # Last sample in high-res window
                # If plotting is enabled, generate response and control effort plots

        if plot:
            ref = np.repeat(reference_vector, samp_per_dt, axis=1)
            loss = [np.mean(np.square(output_conti-ref))]
            index_n_close_loop(
                        loss, None, None,
                        ref, output_conti,
                        time_vector_conti, state_conti,
                        np.repeat(control_effort_matrix, samp_per_dt, axis=1),
                        continuous=True,
                        fig_size=kwargs.get('fig_size', (8, 6))
                        )
        return {
            'output': output_conti,
            'state': state_conti,
            'control_effort': np.repeat(control_effort_matrix, samp_per_dt, axis=1),
            'time_vector': time_vector_conti
        }
