import numpy as np
import pandas as pd
import torch

from histopatseg.constants import CLASS_MAPPING


def get_device(gpu_id=None):
    """Select the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")  # Default to first GPU
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device


def get_class_weights(metadata: pd.DataFrame,
                      task: str = "class_name") -> torch.Tensor:
    """
    Compute normalized class weights directly from a metadata DataFrame.
    The function filters the metadata based on the resolution (e.g., '20x' or '40x'),
    computes the frequency for each class (from the 'task' column),
    and returns a tensor of weights in the order defined by CLASS_MAPPING.
    
    Weights are computed as the inverse frequency and then normalized such that
    the sum of weights equals the number of classes.
    
    Args:
        metadata (pd.DataFrame): Metadata DataFrame with columns including "resolution" and "class_name".
        
    Returns:
        torch.Tensor: 1D tensor of normalized class weights ordered by CLASS_MAPPING.
    """
    # Filter metadata to only include rows with the desired resolution.

    # Total number of samples after filtering
    total = len(metadata)

    # Compute counts for each class defined in CLASS_MAPPING
    # Ensure that we cover all classes even if some counts are zero.
    counts = {cls: (metadata[task] == cls).sum() for cls in CLASS_MAPPING}

    # Order the classes based on the indices in CLASS_MAPPING
    ordered_classes = sorted(CLASS_MAPPING, key=lambda k: CLASS_MAPPING[k])
    count_array = np.array([counts.get(cls, 0) for cls in ordered_classes])

    # Compute frequency (ratio) per class; avoid division by zero
    epsilon = 1e-6  # small constant in case a class is missing
    ratio_array = count_array / total
    ratio_array = np.where(ratio_array == 0, epsilon, ratio_array)

    # Compute raw weights as the inverse of the frequencies
    raw_weights = 1.0 / ratio_array

    # Normalize weights so that the sum equals the number of classes
    num_classes = len(ordered_classes)
    normalized_weights = raw_weights / np.sum(raw_weights) * num_classes

    return torch.FloatTensor(normalized_weights)
