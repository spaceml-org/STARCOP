import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from starcop.models.model_module import ModelModule, differences, pred_classification
import torch
import torchmetrics
import starcop.metrics as starcopmetrics
import starcop.plot as starcoplot
from typing import Optional, List, Dict, Tuple
from numbers import Number
import tempfile
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from starcop.torch_utils import to_device
import pytorch_lightning as pl
from omegaconf import OmegaConf
from kornia.morphology import dilation as kornia_dilation
from kornia.morphology import erosion as kornia_erosion
from starcop.data.normalizer_module import DataNormalizer


def binary_opening(x:torch.Tensor, kernel:torch.Tensor)-> torch.Tensor:
    eroded = torch.clamp(kornia_erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(kornia_dilation(eroded.float(), kernel), 0, 1) > 0



class Mag1cBaseline(pl.LightningModule):
    """
    Uses mag1c output with morphological operations as a baseline
    """
    
    def __init__(self, input_products:List[str], mag1c_threshold:float = 500.0):
        super().__init__()
        self.band_mag1c = input_products.index("mag1c")
        self.mag1c_threshold = mag1c_threshold
        self.element_stronger = torch.nn.Parameter(torch.from_numpy(np.array([[0,1,0],
                                                                              [1,1,1],
                                                                              [0,1,0]])).float(),requires_grad=False)
        
        settings_normalizer = OmegaConf.create({"dataset":{ "input_products": input_products, "output_products": ['labelbinary']}})
        self.normalizer = DataNormalizer(settings_normalizer)

        # self.save_hyperparameters({"settings": {"dataset": {"input_products":["mag1c"], "output_products": ["labelbinary"]}}})

    def forward(self, x:torch.Tensor) -> torch.Tensor:        
        mag1c = x[:, self.band_mag1c:(self.band_mag1c+1)]

        return mag1c
    
    def apply_threshold(self, pred:torch.Tensor, threshold) -> torch.Tensor:
        mag1c_thresholded = (pred > threshold)

        # https://programtalk.com/python-more-examples/kornia.morphology.dilation.bool/
        return binary_opening(mag1c_thresholded, self.element_stronger).long()
        
        
    def batch_with_preds(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = batch.copy()        
        pred = self(batch["input"])
        
        batch["input_norm"] = self.normalizer.normalize_x(batch["input"])
        batch["output_norm"] = self.normalizer.normalize_y(batch["output"])

        batch["prediction"] = pred  # torch.sigmoid((pred - self.mag1c_threshold) / 250)

        

        batch["pred_binary"] = self.apply_threshold(pred, self.mag1c_threshold)
        batch["differences"] = differences(batch["pred_binary"], batch["output_norm"].long())

        batch["pred_classification"] = pred_classification(batch["pred_binary"])

        return batch

    
    
class SanchezBaseline(pl.LightningModule):
    """
    Uses Sanchez ratio with morphological operations as a baseline
    the paper suggested this one as the best: "B8 against the MLR of B1-B6"
    
    Reasonable threshold found to be 0.05
    """
    def __init__(self, input_products:List[str], baseline_threshold:float = 0.05,
                use_normalisation = True, use_morphological_ops = True, band_name = "ratio_wv3_B8_B8MLR_SanchezGarcia22_sum_c_out"):
        super().__init__()
        self.band_baseline = input_products.index(band_name) # note: change band_name and also the input products if you want to run this baseline with different bands setup...
        self.baseline_threshold = baseline_threshold
        self.element_stronger = torch.nn.Parameter(torch.from_numpy(np.array([[0,1,0],
                                                                              [1,1,1],
                                                                              [0,1,0]])).float(),requires_grad=False)
        
        settings_normalizer = OmegaConf.create({"dataset":{ "input_products": input_products, "output_products": ['labelbinary']}})
        self.normalizer = DataNormalizer(settings_normalizer)

        self.use_normalisation = use_normalisation
        self.use_morphological_ops = use_morphological_ops
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:        
        selected_baseline = x[:, self.band_baseline:(self.band_baseline+1)]

        return selected_baseline
    
    def apply_threshold(self, pred:torch.Tensor, threshold) -> torch.Tensor:
        baseline_thresholded = (pred > threshold)

        if self.use_morphological_ops:
            # https://programtalk.com/python-more-examples/kornia.morphology.dilation.bool/
            return binary_opening(baseline_thresholded, self.element_stronger).long()
        else:
            return baseline_thresholded.long()
        
        
    def batch_with_preds(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = batch.copy()        
        
        batch["input_norm"] = self.normalizer.normalize_x(batch["input"])
        batch["output_norm"] = self.normalizer.normalize_y(batch["output"])

        if self.use_normalisation:
            pred = self(batch["input_norm"])
        else:
            pred = self(batch["input"])

        batch["prediction"] = pred  # torch.sigmoid((pred - self.mag1c_threshold) / 250)

        

        batch["pred_binary"] = self.apply_threshold(pred, self.baseline_threshold)
        batch["differences"] = differences(batch["pred_binary"], batch["output_norm"].long())

        batch["pred_classification"] = pred_classification(batch["pred_binary"])

        return batch

    
class VaronBaseline(pl.LightningModule):
    """
    Uses Varon ratio with morphological operations as a baseline
    we found the best ratio to be: "B7 <> B5 ratio_wv3_B7_B5_varon21_sum_c_out"
    
    Reasonable threshold found to be 0.05
    """
    def __init__(self, input_products:List[str], baseline_threshold:float = 0.05,
                use_normalisation = True, use_morphological_ops = True):
        super().__init__()
        self.band_baseline = input_products.index("ratio_wv3_B7_B5_varon21_sum_c_out")
        self.baseline_threshold = baseline_threshold
        self.element_stronger = torch.nn.Parameter(torch.from_numpy(np.array([[0,1,0],
                                                                              [1,1,1],
                                                                              [0,1,0]])).float(),requires_grad=False)
        
        settings_normalizer = OmegaConf.create({"dataset":{ "input_products": input_products, "output_products": ['labelbinary']}})
        self.normalizer = DataNormalizer(settings_normalizer)

        self.use_normalisation = use_normalisation
        self.use_morphological_ops = use_morphological_ops
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:        
        selected_baseline = x[:, self.band_baseline:(self.band_baseline+1)]

        return selected_baseline
    
    def apply_threshold(self, pred:torch.Tensor, threshold) -> torch.Tensor:
        baseline_thresholded = (pred > threshold)

        if self.use_morphological_ops:
            # https://programtalk.com/python-more-examples/kornia.morphology.dilation.bool/
            return binary_opening(baseline_thresholded, self.element_stronger).long()
        else:
            return baseline_thresholded.long()
        
        
    def batch_with_preds(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = batch.copy()        
        
        batch["input_norm"] = self.normalizer.normalize_x(batch["input"])
        batch["output_norm"] = self.normalizer.normalize_y(batch["output"])

        if self.use_normalisation:
            pred = self(batch["input_norm"])
        else:
            pred = self(batch["input"])

        batch["prediction"] = pred  # torch.sigmoid((pred - self.mag1c_threshold) / 250)

        

        batch["pred_binary"] = self.apply_threshold(pred, self.baseline_threshold)
        batch["differences"] = differences(batch["pred_binary"], batch["output_norm"].long())

        batch["pred_classification"] = pred_classification(batch["pred_binary"])

        return batch
