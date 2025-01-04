import numpy as np

def randomized_pulse_generator(end_time, min_time, max_time, sampling_time, num_signal):
    """
    Generates multiple randomized pulse signals, each toggling between 0 and 1
    at random intervals in [min_time, max_time]. The signals are sampled at
    'sampling_time' increments from t=0 to t=end_time.

    Parameters
    ----------
    end_time : float
        Total time duration for the pulse signals.
    min_time : float
        Minimum time interval for toggling each pulse.
    max_time : float
        Maximum time interval for toggling each pulse.
    sampling_time : float
        Time step for the sampled output signal.
    num_signal : int
        Number of independent randomized pulse signals to generate.

    Returns
    -------
    t : np.ndarray of shape (N,)
        Time vector sampled at the given rate, going from 0 to end_time.
    y : np.ndarray of shape (num_signal, N)
        The pulse values for each signal, sampled at the same times in t.
    """

    # 1. Create the time vector
    t = np.arange(0, end_time + 1e-12, sampling_time)
    N = len(t)

    # 2. Initialize each signal to a random binary value (0 or 1)
    current_values = np.random.randint(0, 2, size=num_signal)  # shape (num_signal,)

    # 3. Prepare the output array: (num_signal x N)
    y = np.zeros((num_signal, N), dtype=int)

    # 4. Assign each signal a random "next toggle time" initially
    #    i.e., the time when it will toggle from 0->1 or 1->0
    next_toggle = np.random.uniform(low=min_time, high=max_time, size=num_signal)

    # 5. Step through time and update signals
    for i, current_t in enumerate(t):
        # Record current values at this time step
        y[:, i] = current_values

        # Check which signals should toggle now
        for s in range(num_signal):
            # If we've reached or passed the next toggle time for this signal:
            if current_t >= next_toggle[s]:
                # Toggle it
                current_values[s] = 1 - current_values[s]
                # Pick a new toggle time (random interval from [min_time, max_time])
                next_toggle[s] = current_t + np.random.uniform(min_time, max_time)

    return t, y
