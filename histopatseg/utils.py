import numpy as np
import torch

from histopatseg.constants import CLASS_MAPPING, RATIO_PER_CLASS_10X, RATIO_PER_CLASS_20X


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


def get_class_weights(magnification: int) -> torch.Tensor:
    """
    Compute normalized class weights based on inverse class ratios and return them as a tensor.

    Args:
        magnification (int): The magnification level, either 20 or 10.

    Returns:
        torch.Tensor: A 1D tensor of class weights in the order defined by CLASS_MAPPING.
    """
    # Select the appropriate ratio dictionary.
    if magnification == 20:
        ratios = RATIO_PER_CLASS_20X
    elif magnification == 10:
        ratios = RATIO_PER_CLASS_10X
    else:
        raise ValueError("Unsupported magnification. Choose 20 or 10.")

    # Ensure the classes are processed in the order defined by CLASS_MAPPING.
    ordered_classes = sorted(CLASS_MAPPING, key=lambda k: CLASS_MAPPING[k])

    # Extract the ratio values in the same order.
    ratio_values = np.array([ratios[cls] for cls in ordered_classes])

    # Compute raw weights as the inverse of the ratios.
    raw_weights = 1.0 / ratio_values

    # Normalize weights so that their sum equals the number of classes.
    num_classes = len(ordered_classes)
    normalized_weights = raw_weights / np.sum(raw_weights) * num_classes

    # Return a torch tensor with the weights
    return torch.FloatTensor(normalized_weights)
