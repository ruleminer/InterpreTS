import numpy as np

def calculate_approximate_entropy(data, m=2, r=0.2):
    """
    Calculate the Approximate Entropy (ApEn) of a dataset.

    Parameters
    ----------
    data : array-like
        A 1D array or list of numerical data points.
    m : int, optional
        The length of the pattern to compare (embedding dimension). Default is 2.
    r : float, optional
        The tolerance value for defining similarity between points. Default is 0.2.

    Returns
    -------
    float
        The Approximate Entropy of the dataset. If the dataset is too small, NaN is returned.

    Notes
    -----
    - Approximate Entropy measures the regularity of data. A lower value indicates more regularity.
    - The function uses the method described by Pincus (1991) to calculate ApEn.
    """
    
    N = len(data)
    if N <= m:
        return np.nan

    # Convert data to numpy array for easier manipulation
    data = np.array(data)

    # Define a function to calculate the maximum distance between two vectors
    def max_dist(x, y):
        return np.max(np.abs(x - y))

    # Define a function to calculate the number of similar patterns
    def phi(m):
        X = np.array([data[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                if max_dist(X[i], X[j]) < r:
                    C[i] += 1
        # Ensuring that the argument inside the log is never less than 1
        C = np.maximum(C, 1)  # To avoid log(0) or very small values
        # Adding a very small constant (e.g. 1e-10) to avoid logarithms of zero
        return np.sum(np.log(C / (N - m + 1) + 1e-10))

    # Calculate phi(m) and phi(m+1)
    phi_m = phi(m)
    phi_m_plus_1 = phi(m + 1)

    # Approximate Entropy is the difference between the two
    result = phi_m - phi_m_plus_1

    # If the result is very close to 0, treat it as 0
    if abs(result) < 1e-8:
        result = 0.0
    
    return result
