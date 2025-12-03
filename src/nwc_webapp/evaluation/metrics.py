import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def POD(obs, pred, threshold=0.1):
    """
    POD - Probability of Detection

    POD = TP / (TP + FN) = hits / (hits + misses)

    Measures the fraction of observed events that were correctly predicted.
    Range: [0, 1], where 1 is perfect.

    Args:
        obs (numpy.ndarray): observations
        pred (numpy.ndarray): predictions
        threshold (float): threshold for rainfall values binarization

    Returns:
        float: POD value (None if undefined)
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pred=pred, threshold=threshold)

    if (hits + misses) == 0:
        return None

    return hits / (hits + misses)


def FAR(obs, pred, threshold=0.1):
    """
    FAR - False Alarm Rate

    FAR = FP / (TP + FP) = falsealarms / (hits + falsealarms)

    Measures the fraction of predicted events that did not occur.
    Range: [0, 1], where 0 is perfect (no false alarms).

    Args:
        obs (numpy.ndarray): observations
        pred (numpy.ndarray): predictions
        threshold (float): threshold for rainfall values binarization

    Returns:
        float: FAR value (None if undefined)
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pred=pred, threshold=threshold)

    if (hits + falsealarms) == 0:
        return None

    return falsealarms / (hits + falsealarms)


def CSI(obs, pred, threshold=0.1):
    """
    CSI - critical success index
    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    Args:
        obs (numpy.ndarray): observations
        pred (numpy.ndarray): predictions
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: CSI value
    """

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pred=pred, threshold=threshold)

    if (hits + misses + falsealarms) == 0:
        # print("Error: hits + misses + falsealarms == 0. Returning 0")
        return None

    return hits / (hits + misses + falsealarms)


def prep_clf(obs, pred, threshold=0.1):
    obs = np.where(obs >= threshold, 1, 0)
    pred = np.where(pred >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pred == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pred == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pred == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pred == 0))

    return hits, misses, falsealarms, correctnegatives


def FSS(obs, pred, threshold=0.1, window_size=5):
    """
    FSS - Fractions Skill Score

    Measures spatial similarity between forecast and observation fields
    in terms of fractional coverage above a threshold within a neighborhood.

    Formula: FSS = 1 - (MSE_num / MSE_den)
    where:
        MSE_num = mean((f_pred - f_obs)^2)
        MSE_den = mean(f_pred^2 + f_obs^2)
        f_pred, f_obs are fractional coverages computed via sliding window

    Range: [0, 1], where 1 is perfect.

    Args:
        obs (numpy.ndarray): observations
        pred (numpy.ndarray): predictions
        threshold (float): threshold for rainfall values binarization
        window_size (int): size of the neighborhood window (in pixels)

    Returns:
        float: FSS value (None if undefined)
    """
    # Binarize fields
    obs_binary = (obs >= threshold).astype(np.float32)
    pred_binary = (pred >= threshold).astype(np.float32)

    # Compute fractional coverage using uniform filter (sliding window average)
    # mode='constant' pads with zeros (like PyTorch's avg_pool2d)
    f_obs = uniform_filter(obs_binary, size=window_size, mode='constant')
    f_pred = uniform_filter(pred_binary, size=window_size, mode='constant')

    # Compute MSE components
    mse_num = np.mean((f_pred - f_obs) ** 2)
    mse_den = np.mean(f_pred ** 2 + f_obs ** 2)

    # Handle edge case: no coverage in both obs and pred
    if mse_den == 0:
        if mse_num == 0:
            # Perfect match (both empty)
            return 1.0
        else:
            # Unusual case
            return None

    fss_value = 1 - (mse_num / mse_den)

    # Clamp to [0, 1] range
    return np.clip(fss_value, 0.0, 1.0)


def compute_CSI(targets, outputs, thresholds=None):
    if thresholds is None:
        thresholds = [1, 5, 10, 20, 50]

    prediction_times = list(targets.keys())
    prediction_offsets = [f"{5 * (i + 1)}_min" for i in range(len(prediction_times))]

    data = pd.DataFrame(index=thresholds, columns=prediction_offsets)

    for idx, pred_time in enumerate(prediction_times):
        for th in thresholds:
            metric_value = CSI(targets[pred_time], outputs[pred_time], threshold=th)

            data.at[th, prediction_offsets[idx]] = metric_value

    data = data.fillna(value=0)
    return data
