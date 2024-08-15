import  numpy as np

def create_line(slope, intercept):
    """Return x and y arrays that express a line with slope and intercept."""
    x_vals = np.array([min(x), max(x)])
    y_vals = intercept + slope * x_vals
    return x_vals, y_vals


# Define the model function for the surface
def model_surface(x, y, a, b, bias):
    return a * x + b * y + bias
    
def calculate_bce_loss(y_true, y_pred):
    """
    Calculate the binary cross-entropy loss.

    Parameters:
    -----------
    y_true (array-like): True binary labels (0 or 1).
    y_pred (array-like): Predicted probabilities, between 0 and 1.

    Returns:
    --------
    float: The average binary cross-entropy loss.
    """
    # Ensure that y_pred does not contain values exactly equal to 0 or 1,
    # as log(0) is undefined and can cause computation errors.
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss