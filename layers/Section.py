import numpy as np
from visualizations.plot_metrics import plot_metrics
from IPython.display import clear_output
from layers.Dropout import Dropout
import matplotlib.pyplot as plt


#############################################################################################################################

class NN_section:
    """
    Compile and manage the training and inference process of a neural network model.

    Parameters:
    -----------
    model : list
        A list of layers (objects) that define the structure of the neural network. Each layer should implement 
        a forward and backward method for inference and backpropagation, respectively.

    Attributes:
    -----------
    model : list
        The list of layers comprising the neural network.
    """

    def __init__(self, model: list):
        """
        Initialize the compilation class with the given model.

        Parameters:
        -----------
        model : list
            A list containing the layers of the model in the order of forward propagation.
        """
        # Validate layer compatibility: Check if each layer's output size matches the next layer's input size
        for i in range(len(model) - 1):

            # Set Dropout undefined attributes
            if isinstance(model[i+1], Dropout):
                model[i+1].input_size = model[i].output_size
                model[i+1].output_size = model[i+1].input_size
                model[i+1].batch_size = np.inf

            # Check for residuals size compability
            if model[i].output_size != model[i + 1].input_size:
                raise ValueError(
                    f"Layer mismatch detected: Layer {i + 1} output size ({model[i].output_size}) "
                    f"does not match Layer {i + 2} input size ({model[i + 1].input_size})."
                )
        self.batch_size = min(layer.batch_size for layer in model)
        self.model = model
        self.input_size = model[0].input_size
        self.output_size = model[-1].output_size
        self.batch_size = min(layer.batch_size for layer in model)
        self.activation = 'not defined for this'

    #################################################################

    def reset_memory(self):
        for layer in self.model:
            try:
                layer.reset_memory()
            except:
                pass

    #################################################################

    def reset_Loss_function(loss_function):
        if hasattr(loss_function, 'memory'):
            loss_function.reset_memory()

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the compiled model.

        This method iterates over all the layers in the model and sums up the number 
        of trainable parameters from each layer.

        Returns:
        int: The total number of trainable parameters in the model.
        """
    
        # Initialize a variable to keep track of the total number of parameters.
        params = 0

        # Iterate through each layer in the model
        for layer in self.model:
            # Add the number of trainable parameters from each layer
            params += layer.trainable_params()
        
        # Return the total count of trainable parameters
        return int(params)
    
    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the compiled model.

        This method iterates over all the layers in the model and sums up the number 
        of parameters from each layer.

        Returns:
        int: The total number of parameters in the model.
        """
    
        # Initialize a variable to keep track of the total number of parameters.
        params = 0

        # Iterate through each layer in the model
        for layer in self.model:
            # Add the number of parameters from each layer
            params += layer.all_params()
        
        # Return the total count of trainable parameters
        return int(params)
    
    #################################################################

    def summary(self) -> None:
        """
        Prints a detailed summary of the model's architecture, including the number of parameters (trainable
        and non-trainable) for each layer, along with activation functions and layer types.
        
        Returns
        -------
        None
            This method only outputs the model summary to the console.
        """
        # Print the title of the model summary with decorative asterisks
        print('\n', '*' * 30, 'model summary', '*' * 30, '\n')
        
        # Initialize counters for the total number of trainable and all parameters in the model
        total_n_trainable = 0
        total_n_all = 0
        
        # Iterate through each layer in the model and gather information for summary
        for index, layer in enumerate(self.model):
            # Print layer index (1-based) and type of layer
            print(f'layer {index+1}:', end='\n\t')
            print(type(layer), end='\n\t')
            
            # Print the activation function used in the current layer
            print('activation function:', layer.activation, end='\n\t')
            # Print batch size of model, input size and neuron numbers
            print('batch size:', layer.batch_size, end='\n\t')
            print('input size:', layer.input_size, end='\n\t')
            print('output size:', layer.output_size, end='\n\t')
            
            # Get the number of trainable parameters for the current layer
            n_trainable = layer.trainable_params()
            # Accumulate the trainable parameter count to the total
            total_n_trainable += n_trainable
            
            # Get the total number of parameters (trainable + non-trainable) in the current layer
            n_all = layer.all_params()
            # Accumulate the total parameter count
            total_n_all += n_all
            
            # Print the total number of parameters in the current layer
            print(f'number of parameters: {n_all}', end='\n\t')
            
            # Print the number of trainable parameters in the current layer
            print(f'number of trainable parameters: {n_trainable}', end='\n\t')
            
            # Calculate and print the number of non-trainable parameters in the current layer
            print(f'number of non trainable parameters: {n_all - n_trainable}', end='\n\t')
            
            # Print a separator line for clarity between layers
            print('-' * 50)
        
        # Print the total number of parameters across all layers in the model
        print(f'total number of parameters: {total_n_all}', end='\n\t')
        
        # Print the total number of trainable parameters across the model
        print(f'total number of trainable parameters: {total_n_trainable}', end='\n\t')
        
        # Print the total number of non-trainable parameters across the model
        print(f'total number of non trainable parameters: {total_n_all - total_n_trainable}', end='\n\t')

    #################################################################

    def validate_error_nn(self, error_nn: np.ndarray) -> None:
        """
        Validates the error of the final layer to ensure it matches the expected output size and dimensions.

        Parameters:
        -----------
        error_nn : np.ndarray
            The output array from the final layer to validate.

        Raises:
        -------
        ValueError:
            If the error_nn size does not match the expected output size of the last layer.
            If the error dimension is less than 2.
        """
        # Ensure the output has at least 2 dimensions (batch size and output size)
        if error_nn.ndim < 2:
            raise ValueError(
                f"Invalid error dimensions: The final layer's error must have at least 2 dimensions, "
                f"but got {error_nn.ndim} dimensions."
            )

        # Retrieve the expected output size of the last layer
        final_layer_output_size = self.model[-1].output_size

        # Determine the expected output shape, handling both int and tuple cases
        if isinstance(final_layer_output_size, int):
            expected_error_shape = (final_layer_output_size,)
        else:
            expected_error_shape = final_layer_output_size

        # Validate that the provided output size matches the expected output size of the last layer
        if error_nn.shape[1:] != expected_error_shape:
            raise ValueError(
                f"Last layer's error array size mismatch: Model produced error of size {error_nn.shape[1:]}, "
                f"which does not match the expected error size {expected_error_shape} "
                "of the final layer."
            )

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Update the parameters of all layers using the provided gradients.

        Parameters:
        -----------
        grads : np.ndarray
            The gradients for all trainable parameters in the model.
        learning_rate : float, optional
            The learning rate for parameter updates (default is 1e-3).

        Returns:
        --------
        None
        """
        ind2 = 0
        for layer in self.model:
            ind1 = ind2
            ind2 += layer.trainable_params()
            layer.update(grads=grads[ind1:ind2], learning_rate=learning_rate)

    #################################################################

    def optimizer_init(self, method='Adam', **kwargs):
        # Initialize optimizer for each layer if has not been defined yet or has changed
        if not hasattr(self, 'Optimizer'):
            self.Optimizer = method
            for layer in self.model:
                layer.optimizer_init(optimizer=method, **kwargs)
        if self.Optimizer != method:
            for layer in self.model:
                layer.optimizer_init(optimizer=method, **kwargs)

    #################################################################

    def backward(self, input: np.ndarray, error_nn: np.ndarray, batch_size: int = 8, modify: bool = True,
                 learning_rate: float = 1e-3, return_error: bool = False, shuffle: bool = False,
                 return_grads: bool = False) -> None:
        """
        Perform the backpropagation step to update the model's weights.

        Parameters:
        -----------
        input : np.ndarray
            Input data for the current batch.
        error_nn : np.ndarray
            Error propagated to last layer
        batch_size : int, optional
            Training batch_size
        modify : bool
            If false, model would not been update
        learning_rate : float, optional
            Learning rate for updating weights (default is 1e-3).
        modify : bool, optional
            If you dont want to modify layers, but only want to calculate backward error, set this argument to False
        return_error : bool, optional
            return derivatives with respect to input for whole batch
        shuffle : bool, optionaal
            Shuffle data before training process
        return_grads : bool
            return gradients with respect to parameters
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(input)
        self.validate_error_nn(error_nn)

        # Shuffle data if needed
        if shuffle:
            random_indices = np.random.permutation(input.shape[0])
            input = input[random_indices]
            error_nn = error_nn[random_indices]
        # Determine how many batches are needed based on batch size
        batch_num = int(np.ceil(input.shape[0] / batch_size))

        # Derivatives with respect to netwrok input
        if return_error:
            error_in = np.zeros(input.shape)

        grads = np.zeros((self.trainable_params(), 1))

        # Iterate over each batch
        for i in range(batch_num):
            # Extract the current batch of input data and corresponding targets
            data_X = input[i * batch_size: (i + 1) * batch_size].copy()
            error = error_nn[i * batch_size: (i + 1) * batch_size].copy()

            # Perform forward pass to get the output
            self.reset_memory()
            _ = self(data_X)

            # Perform backpropagation on each layer, starting from the last layer
            batch_grad = np.array([]).reshape((-1,1))
            for layer in reversed(self.model):
                # print(layer.input.shape)
                x = layer.backward(error, learning_rate=learning_rate, return_error=True, modify=False, return_grads=True)
                error = x['error_in']
                grad = x['gradients']
                if grad is not None:
                    batch_grad = np.concatenate((grad, batch_grad), axis=0)
            grads += batch_grad * batch_size
            
            # accumulate error_in if necessarry
            if return_error:
                error_in[i * batch_size: (i + 1) * batch_size] = error_in
        
        if modify:
            self.update(grads, learning_rate=learning_rate)
            
        # Return error propagated to possible pervious layer
        if return_error and not return_grads:
            return error_in
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        if return_grads:
            return grads
        else:
            return None

    #################################################################

    def validate_input(self, X: np.ndarray) -> None:
        """
        Validates the input to ensure it matches the expected dimensions and size for the first layer.

        Parameters:
        -----------
        input : np.ndarray
            The input array to validate.

        Raises:
        -------
        ValueError:
            If the input does not have at least 2 dimensions (batch size and input size).
            If the input size does not match the first layer's expected input size.
        """
        # Check if input has at least 2 dimensions (batch size and input size)
        if X.ndim < 2:
            raise ValueError(
                f"Input dimension is too low: Expected at least 2 dimensions (batch size and input size), "
                f"but got {X.ndim} dimensions."
            )

        # Retrieve the first layer's expected input size
        first_layer_input_size = self.model[0].input_size

        # Determine the expected input shape, handling both int and tuple cases
        if isinstance(first_layer_input_size, int):
            expected_input_shape = (first_layer_input_size,)
        else:
            expected_input_shape = first_layer_input_size

        # Validate that the provided input size matches the first layer's expected input size
        if X.shape[1:] != expected_input_shape:
            raise ValueError(
                f"Input size mismatch: Provided input size {X.shape[1:]} "
                f"does not match the first layer's expected input size {expected_input_shape}."
        )

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the model.

        Parameters:
        -----------
        input : np.ndarray
            The input data to be fed into the model.

        Returns:
        --------
        np.ndarray:
            The output of the model after processing the input.
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(input)

        # Determine how many batches are needed based on batch size
        batch_num = int(np.ceil(input.shape[0] / self.batch_size))

        shape = (input.shape[0], ) + self.model[-1].output_size if self.model[-1].output_size is tuple\
            else (input.shape[0], self.model[-1].output_size)
        out = np.zeros(shape)

        # Process each batch of data
        for i in range(batch_num):
            # Get the batch of input data
            data_X = input[i * self.batch_size: (i + 1) * self.batch_size]
            layer_in = data_X.copy()

            # Forward pass through each layer of the model
            for layer in self.model:
                layer_in = layer(layer_in)
            
            out[i * self.batch_size: (i + 1) * self.batch_size] = layer_in

        # Reshape and return the final output
        return out

    #################################################################

    def Jaccobian(self, X_train: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for the model given training input data.

        Parameters:
        -----------
        X_train : np.ndarray
            Training data with shape (n_samples, input_size).

        Returns:
        --------
        jaccob : np.ndarray
            The Jacobian matrix with shape (n_samples * output_size, trainable_params).
            Represents partial derivatives of each output with respect to trainable parameters.
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)

        if type(self.model[-1].output_size) is not int:
            raise TypeError('Jaccobian calculations for the last layer is not supported, use reshaping to see what would happen')
        # Initialize an empty Jacobian matrix with shape (total outputs, total trainable parameters)
        jaccob = np.zeros((X_train.shape[0] * self.model[-1].output_size, self.trainable_params()))

        # Loop over each training sample
        for batch_index in range(X_train.shape[0]):
            # Loop over each output neuron
            for out_ind in range(self.model[-1].output_size):
                # Initialize a vector to isolate one output at a time for differentiation
                E_neuron = np.zeros((1, self.model[-1].output_size))
                E_neuron[:, out_ind] = 1.0
                
                # Temporary variable to store the Jacobian row
                J_row = np.array([])

                # Perform a forward pass with the current input
                _ = self(X_train[batch_index].reshape((1, -1)))
                
                # Backpropagate through each layer in reverse order
                for layer in reversed(self.model):
                    if layer == self.model[0]:  # If the layer is the first one
                        # Get gradients for the first layer without modifying the layer state
                        grads = layer.backward(E_neuron, return_grads=True, modify=False)
                        # Concatenate gradients into the Jacobian row
                        J_row = np.concatenate((np.ravel(grads), J_row))
                    else:  # For all other layers
                        # Perform backpropagation to get errors and gradients
                        back_out = layer.backward(E_neuron, return_error=True, return_grads=True, modify=False)
                        # Update the error for the next layer
                        E_neuron = back_out['error_in']
                        # Get the gradients for the current layer
                        grads = back_out['gradients']
                        # Concatenate gradients into the Jacobian row
                        J_row = np.concatenate((np.ravel(grads), J_row))
                
                # Update the corresponding row in the Jacobian matrix
                ind = batch_index * self.model[-1].output_size + out_ind
                jaccob[ind] = J_row

        return jaccob

    #################################################################

    def levenberg_mar(self, X_train: np.ndarray, error_nn: np.ndarray,
                      learning_rate: float = 0.7, gamma: float = 0.99) -> dict:
        """
        Perform the Levenberg-Marquardt optimization algorithm for model training.

        Parameters:
        -----------
        X_train : np.ndarray
            Training input data with shape (n_samples, input_size).
            
        error_nn : np.ndarray
            Error propagated to last layer
            
        learning_rate : float, optional
            The learning rate for weight updates. Default is 0.7.
            
        gamma : float, optional
            The damping parameter for the Levenberg-Marquardt algorithm. Default is 0.8.
            
        **kwargs : dict
            Additional keyword arguments for plotting or other purposes.
        """
        if type(self.model[-1].output_size) is not int:
            raise TypeError('Jaccobian calculations for the last layer is not supported, use reshaping to see what would happen')
        
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)
        self.validate_error_nn(error_nn)
        
        # Initialize each layer's optimizer to SGD to avoid momentum effects (e.g., Adam)
        for layer in self.model:
            layer.optimizer_init('SGD')

        # Compute the Jacobian matrix for the current training data
        self.reset_memory()
        J = self.Jaccobian(X_train)

        # Compute the gradient update using the Levenberg-Marquardt formula
        new_grads = np.linalg.inv((J.T @ J + gamma * np.eye(self.trainable_params()))) @ J.T @ error_nn.reshape((-1, 1))

        # Update model weights using the computed gradients
        ind2 = 0
        for layer in self.model:
            ind1 = ind2
            ind2 += layer.trainable_params()
            layer.update(new_grads[ind1:ind2].reshape((-1, 1)), learning_rate)

    #################################################################

    def gauss_newton(self, X_train: np.ndarray, error_nn: np.ndarray,
                      learning_rate: float = 0.7) -> dict:
        """
        Perform the Gauss-Newton optimization algorithm for model training.

        Parameters:
        -----------
        X_train : np.ndarray
            Training input data with shape (n_samples, input_size).
            
        error_nn : np.ndarray
            Error propagated to last layer
            
        learning_rate : float, optional
            The learning rate for weight updates. Default is 0.7.
            
        **kwargs : dict
            Additional keyword arguments for plotting or other purposes.
        """
        if type(self.model[-1].output_size) is not int:
            raise TypeError('Jaccobian calculations for the last layer is not supported, use reshaping to see what would happen')
        
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)
        self.validate_error_nn(error_nn)
        
        # Initialize each layer's optimizer to SGD to avoid momentum effects (e.g., Adam)
        for layer in self.model:
            layer.optimizer_init('SGD')

        # Compute the Jacobian matrix for the current training data
        self.reset_memory()
        J = self.Jaccobian(X_train)

        # Compute the gradient update using the Levenberg-Marquardt formula
        try:
            new_grads = np.linalg.inv(J.T @ J) @ J.T @ error_nn.reshape((-1, 1))
        except:
            new_grads = np.linalg.inv(J.T @ J + 0.1 * np.eye(self.trainable_params())) @ J.T @ error_nn.reshape((-1, 1))

        # Update model weights using the computed gradients
        ind2 = 0
        for layer in self.model:
            ind1 = ind2
            ind2 += layer.trainable_params()
            layer.update(new_grads[ind1:ind2].reshape((-1, 1)), learning_rate)