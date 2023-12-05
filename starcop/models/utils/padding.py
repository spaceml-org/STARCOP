import numpy as np
import torch
from numpy.typing import ArrayLike

def find_padding(v:int, divisor:int=8):
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def padded_predict(tensor:ArrayLike, model:torch.nn.Module, divisor:int=32,
                   device:torch.device=torch.device("cpu")) -> ArrayLike:
    """
    Predict on a tensor adding padding if necessary

    Args:
        tensor: np.array (C, H, W) of input values
        model: torch.nn.Module
        divisor: int
        device: torch.device
    
    Returns:
        2D or 3D np.array with the prediction
    """
    assert len(tensor.shape) == 3, f"Expected 3D tensor, found {len(tensor.shape)}D tensor"

    pad_r = find_padding(tensor.shape[-2], divisor)
    pad_c = find_padding(tensor.shape[-1], divisor)

    tensor_padded = np.pad(
        tensor, ((0, 0), (pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), "reflect"
    )

    slice_rows = slice(pad_r[0], None if pad_r[1] <= 0 else -pad_r[1])
    slice_cols = slice(pad_c[0], None if pad_c[1] <= 0 else -pad_c[1])

    tensor_padded = torch.tensor(tensor_padded, device=device)[None]  # Add batch dim

    with torch.no_grad():
        pred_padded = model(tensor_padded)[0]
        if len(pred_padded.shape) == 3:
            pred = pred_padded[(slice(None), slice_rows, slice_cols)]
        elif len(pred_padded.shape) == 2:
            pred = pred_padded[(slice_rows, slice_cols)]
        else:
            raise NotImplementedError(f"Don't know how to slice the tensor of shape {pred_padded.shape}")


    return np.array(pred.cpu())