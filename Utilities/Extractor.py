# -*- coding: utf-8 -*-

"""
@author: Eslam Mohamed
@email: es.ahmed@nu.edu.eg

Functions to read, load and process EEG Data
"""

import os
import pandas as pd
import numpy as np
from scipy import signal
from typing import List, Tuple

def process_eeg(TIME_STEPS:int = 1200, included_states: List[str] =["Up", "Down", "Left", "Right", "Select"], subject_folder :str ='/kaggle/input/arabic-eeg-sessions/RecordedSessions/Antony')->Tuple[np.ndarray, np.array]:
    """
        Process EEG data files from a specified subject folder and extract relevant EEG data segments.

        Parameters:
        TIME_STEPS (int): The number of time steps for each EEG data segment. Default is 1200.
        included_states (list): List of states to include in the processing. Default is ["Up", "Down", "Left", "Right", "Select"].
        subject_folder (str): Path to the folder containing EEG data files for the subject. Default is '/kaggle/input/arabic-eeg-sessions/RecordedSessions/Antony'.

        Returns:
        tuple: A tuple containing:
            - X (np.ndarray): A NumPy array of shape (number of samples, number of EEG channels, TIME_STEPS) containing the processed EEG data.
            - Y (np.ndarray): A NumPy array of shape (number of samples,) containing the corresponding states for each EEG data segment.

        Notes:
        - The function reads CSV files from the specified subject folder.
        - Each CSV file is expected to contain columns 'EEG 1' to 'EEG 8' for EEG channels and 'State' for the state labels.
        - The function groups the EEG data by state transitions and extracts segments of length TIME_STEPS.
        - If a segment is shorter than TIME_STEPS, it is padded with zeros.
        - The function ensures that all extracted segments have the same shape.
        - If there are inconsistent shapes, the function filters out those segments and only retains the consistent ones.
    """

    files = os.listdir(subject_folder)
    dfs = []

    # Read and process each file
    for subject, file in enumerate(files):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(subject_folder, file))
            df['Subject'] = subject + 1
            dfs.append(df[['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'State', 'Subject']])
        else:
            #Skipping non-CSV file
            continue

    # Process EEG data for each state
    all_state_data = []

    for df in dfs:
        state_groups = df.groupby((df['State'] != df['State'].shift()).cumsum())

        for _, data in state_groups:
            state = data['State'].iloc[0]
            if state in included_states:
                eeg_data = np.transpose(data[['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8']].values)[:,:TIME_STEPS]
                # apply padding if timesteps are smaller than 1200
                if eeg_data.shape[1] < TIME_STEPS:
                    pad_width = TIME_STEPS - eeg_data.shape[1]
                    eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                else:
                    eeg_data = eeg_data[:, :TIME_STEPS]

                all_state_data.append(pd.DataFrame({'State': [state], 'EEG Data': [eeg_data]}))

    # Concatenate the processed data
    final_df = pd.concat(all_state_data, ignore_index=True)

    # Fetch the list of arrays
    data_list = final_df['EEG Data'].values
    state_list = final_df['State'].values

    # Check the shapes of all arrays
    shapes = [arr.shape for arr in data_list]

    # Ensure all shapes are the same
    if len(set(shapes)) == 1:
        # All arrays have the same shape, so convert the list to a NumPy array
        X = np.array([item for item in data_list])
        Y = np.array([state for state in final_df['State'].values])
    else:
        # Print the shapes that are inconsistent
        print("Inconsistent shapes found:", set(shapes))
        # Filter and store only consistent shapes of data
        X = np.array([item for item in data_list if item.shape[1] == TIME_STEPS])
        Y = np.array([state for item, state in zip(data_list, state_list) if item.shape[1] == TIME_STEPS])

    X = X.astype('float64')

    return X, Y


def filter_eeg(X: np.ndarray, sampling_freq: int = 250, notch_freq: float = 60.0, lowcut: float = 0.5, highcut: float = 30.0, scaling_factor: float = 50 / 1e6) -> np.ndarray:
    """
    Filter EEG data using notch and bandpass filters.

    Parameters:
    - X (np.ndarray): EEG data to be filtered with shape (trials, channels, samples).
    - sampling_freq (int, optional): Sampling frequency of the EEG data in Hz. Default is 250 Hz.
    - notch_freq (float, optional): Notch filter frequency in Hz (e.g., 60 Hz for powerline interference). Default is 60 Hz.
    - lowcut (float, optional): Low cutoff frequency for the bandpass filter in Hz. Default is 0.5 Hz.
    - highcut (float, optional): High cutoff frequency for the bandpass filter in Hz. Default is 30 Hz.
    - scaling_factor (float, optional): Scaling factor to convert the filtered data to µV (microvolts). Default is 50 µV / 1e6.

    Returns:
    - np.ndarray: Filtered EEG data with shape (trials, channels, samples) after applying notch and bandpass filters and scaling.

    Notes:
    - The notch filter is designed to remove powerline interference at the specified notch frequency.
    - The bandpass filter is designed to retain frequencies within the specified lowcut and highcut range.
    - The filtered EEG data is scaled to the specified scaling factor.
    """
    # Design the notch filter
    Q = 30  # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs=sampling_freq)

    # Design the bandpass filter
    nyquist_freq = 0.5 * sampling_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b_bandpass, a_bandpass = signal.butter(4, [low, high], btype='band')

    # Initialize filtered EEG data array
    filtered_eeg_data = np.zeros_like(X)

    # Apply the notch and bandpass filters to each trial and channel
    for trial in range(X.shape[0]):
        for channel in range(X.shape[1]):
            # Apply notch filter
            eeg_notch_filtered = signal.filtfilt(b_notch, a_notch, X[trial, channel, :])
            # Apply bandpass filter
            filtered_eeg_data[trial, channel, :] = signal.filtfilt(b_bandpass, a_bandpass, eeg_notch_filtered)

    # Scale the filtered EEG data to ±50 µV
    filtered_eeg_data *= scaling_factor

    return filtered_eeg_data
