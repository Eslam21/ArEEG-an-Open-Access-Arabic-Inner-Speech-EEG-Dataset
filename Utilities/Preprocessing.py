# -*- coding: utf-8 -*-

"""
| Authors        | Emails                      |
|----------------|-----------------------------|
| Youssef Radwan | yo.radwan@nu.edu.eg         |
| Eslam Mohamed  | es.ahmed@nu.edu.eg          |

Utilities to apply preprocessing techniques for EEG data.
"""

from sklearn.decomposition import PCA, FastICA
import numpy as np
import pandas as pd
from typing import Optional, Literal
import warnings

def apply_PCA(X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """
    Apply PCA to EEG data treating channels and time steps as features for each trial.
    Fits PCA on the entire dataset and applies the transformation.
    Maintains the trial axis, flexible on other dimensions based on n_components.

    Parameters:
    - X: NumPy array of shape (trials, channels, time_steps)
    - n_components: Number of principal components to keep. If None, all components are kept.

    Returns:
    - pca_data: Transformed data with shape 
            (trials, n_components) if n_components is specified, 
            else shape will vary based on PCA transformation.
    """
    num_trials = X.shape[0]
    # Flatten channels and time_steps for PCA input
    X_reshaped = X.reshape(num_trials, -1)  # Now (trials, channels*time_steps)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X_reshaped)  # Fit and transform on the entire dataset
    
    return pca_data


def apply_ICA(X: np.ndarray, n_components: Optional[int] = 2) -> np.ndarray:
    """
    Apply ICA to each trial in the data and flatten the output to create feature vectors.
    Fits ICA on the entire dataset and applies the transformation.
    
    Parameters:
    - X: NumPy array of shape (trials, channels, time_steps)
    - n_components: Number of ICA components to extract. If None, all components are used.
    
    Returns:
    - ica_df: Transformed and flattened, where each row is a flattened feature vector for an trial.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Prepare the data for ICA: shape should be (n_samples, n_features)
    num_trials, num_channels, num_time_steps = X.shape
    
    # Reshape and apply ICA on the entire dataset
    X_reshaped = X.reshape(num_trials, num_channels * num_time_steps)
    ica = FastICA(n_components=n_components, random_state=0)
    ica_data = ica.fit_transform(X_reshaped)  # Fit and transform on the entire dataset
    
    return ica_data


def apply_moving_avrg(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Compute the moving average of EEG data along the last axis.

    Parameters:
    - data: NumPy array of shape (experiments, channels, time_steps)
        The EEG data to be processed.
    - window_size: int
        The size of the moving window.

    Returns:
    - result: NumPy array of shape (experiments, channels, time_steps - window_size + 1)
        The result of applying the moving average.
    """
    cumsum = np.cumsum(data, axis=-1)
    cumsum[:, :, window_size:] = cumsum[:, :, window_size:] - cumsum[:, :, :-window_size]
    return cumsum[:, :, window_size - 1:] / window_size



def apply_weighted_moving_avrg(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Compute the weighted moving average of EEG data along the last axis.

    Parameters:
    - data: NumPy array of shape (experiments, channels, time_steps)
        The EEG data to be processed.
    - window_size: int
        The size of the moving window.

    Returns:
    - result: NumPy array of shape (experiments, channels, time_steps - window_size + 1)
        The result of applying the weighted moving average.
    """
    weights = np.arange(1, window_size + 1)
    
    def apply_wma(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, weights / weights.sum(), mode='valid')
    
    return np.apply_along_axis(apply_wma, -1, data)


def compute_statistical_features(data: np.ndarray, axis: Literal['time', 'channels'] = 'time') -> pd.DataFrame:
    """
    Compute statistical features of EEG data either over the time steps for each channel or over channels for each time step.

    Parameters:
    - data: NumPy array of shape (experiments, channels, time_steps)
        The EEG data to be processed.
    - axis: str, optional (default='time')
        Axis over which to compute the features. 'time' computes features over the time steps for each channel,
        'channels' computes features over the channels for each time step.

    Returns:
    - df: pandas DataFrame
        DataFrame containing the computed statistical features (mean, median, rms, total_sum, energy)
        for each channel of each experiment if axis='time', or for each time step if axis='channels'.
    """
    if axis == 'time':
        mean = np.mean(data, axis=2)      # Mean over the time steps for each channel
        median = np.median(data, axis=2)  # Median over the time steps for each channel
        rms = np.sqrt(np.mean(data**2, axis=2))  # Root mean square over the time steps for each channel
        total_sum = np.sum(data, axis=2)  # Sum over the time steps for each channel
        energy = np.sum(data**2, axis=2)  # Energy (sum of squares) over the time steps for each channel
        num_features = data.shape[1]
        columns = [f"{feature}_{i+1}" for feature in ['mean', 'median', 'rms', 'total_sum', 'energy'] for i in range(num_features)]
    elif axis == 'channels':
        mean = np.mean(data, axis=1)      # Mean over the channels for each time step
        median = np.median(data, axis=1)  # Median over the channels for each time step
        rms = np.sqrt(np.mean(data**2, axis=1))  # Root mean square over the channels for each time step
        total_sum = np.sum(data, axis=1)  # Sum over the channels for each time step
        energy = np.sum(data**2, axis=1)  # Energy (sum of squares) over the channels for each time step
        num_features = data.shape[2]
        columns = [f"{feature}_{i+1}" for feature in ['mean', 'median', 'rms', 'total_sum', 'energy'] for i in range(num_features)]
    else:
        raise ValueError("Invalid axis value. Use 'time' or 'channels'.")

    # Combine all features into a single array
    stat_features = np.stack((mean, median, rms, total_sum, energy), axis=2 if axis == 'time' else 1)

    # Reshape stat_features to (experiments, num_features * num_channels or num_time_steps)
    stat_features_reshaped = stat_features.reshape(data.shape[0], -1)

    # Create DataFrame
    df = pd.DataFrame(stat_features_reshaped, columns=columns)

    return df
