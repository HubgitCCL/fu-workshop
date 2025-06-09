import numpy as np
import torch
from typing import List, Dict, Optional

def fedavg(weights_list: List[Dict[str, np.ndarray]], 
           sample_sizes: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Federated averaging of model weights.
    Args:
        weights_list: List of state dictionaries from clients (numpy arrays).
        sample_sizes: List of sample counts for each client. If None, uses equal weights.
    Returns:
        Dictionary of averaged weights.
    """
    if not weights_list:
        raise ValueError("Empty weights list")
    
    n_clients = len(weights_list)

    # Use equal weights if sample_sizes not provided
    if sample_sizes is None:
        sample_sizes = [1] * n_clients
    elif len(sample_sizes) != n_clients:
        raise ValueError("Mismatched number of weights and sample sizes")

    # Convert all client weights to float32 to avoid dtype conflicts
    weights_list_float = []
    for client_weights in weights_list:
        float_weights = {}
        for k, v in client_weights.items():
            float_weights[k] = v.astype(np.float32)
        weights_list_float.append(float_weights)
    weights_list = weights_list_float

    # Calculate total samples and mixing coefficients for weighted average
    total_samples = sum(sample_sizes)
    mixing_coeffs = [size / total_samples for size in sample_sizes]

    # Initialize global weights as zeros (float32) using first client's weights as template
    global_weights = {}
    first_client = weights_list[0]
    for k, v in first_client.items():
        global_weights[k] = np.zeros_like(v, dtype=np.float32)

    # Identify BatchNorm buffer keys that require equal weighting
    bn_keys = ["running_mean", "running_var", "num_batches_tracked"]

    # Aggregate weights
    for key in global_weights:
        if any(bn_key in key for bn_key in bn_keys):
            # Equal weighting for BatchNorm running stats
            for client_weights in weights_list:
                global_weights[key] += client_weights[key] / n_clients
        else:
            # Weighted by sample size for other parameters
            for i, client_weights in enumerate(weights_list):
                coef = mixing_coeffs[i]
                global_weights[key] += client_weights[key] * coef

    return global_weights


def get_model_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """
    Extract model parameters and buffers as numpy arrays.
    """
    weights = {}
    # Include model parameters
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    # Include buffers (e.g., BatchNorm running stats)
    for name, buf in model.named_buffers():
        weights[name] = buf.detach().cpu().numpy()
    return weights


def set_model_weights(model: torch.nn.Module, weights: Dict[str, np.ndarray], device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Assign numpy weight arrays to model parameters and buffers.
    Converts dtype if mismatch with model.
    """
    # Set model parameters
    for name, param in model.named_parameters():
        if name in weights:
            weight_tensor = torch.from_numpy(weights[name])
            # Move tensor to correct device: given device or parameter's device
            if device is not None:
                weight_tensor = weight_tensor.to(device)
            else:
                weight_tensor = weight_tensor.to(param.device)
            # Ensure correct dtype
            if weight_tensor.dtype != param.data.dtype:
                weight_tensor = weight_tensor.to(param.data.dtype)
            param.data = weight_tensor

    # Set buffers (e.g., running_mean, running_var, num_batches_tracked)
    for name, buf in model.named_buffers():
        if name in weights:
            weight_tensor = torch.from_numpy(weights[name])
            if device is not None:
                weight_tensor = weight_tensor.to(device)
            else:
                weight_tensor = weight_tensor.to(buf.device)
            if weight_tensor.dtype != buf.dtype:
                weight_tensor = weight_tensor.to(buf.dtype)
            buf.copy_(weight_tensor)

    return model