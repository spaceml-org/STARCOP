import torch
from typing import Union,Dict
from torch import Tensor

def to_device(x : Union[Dict[str, Tensor], Tensor], device : torch.device) -> Union[Dict[str, Tensor], Tensor]:
    if torch.is_tensor(x):
        return x.to(device)
    elif hasattr(x, "keys"):
        return {k:to_device(v, device) for k, v in x.items()}
    
    return x