from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from torch.utils.data import Dataset
import os
import rasterio
import torch
import rasterio.windows
import numpy as np
import math

class STARCOPEMITDataset(Dataset):
    def __init__(self, dataframe_substitute,
                 input_products:List[str],
                 output_products: List[str],
                 weight_loss:Optional[str]=None,
                 spatial_augmentations=None,
                 extra_products:Optional[List[str]]=None,
                 window_size_sample:Optional[Tuple[int, int]]=None,
                 hyperparams={}):
        
        self.dataframe_substitute = dataframe_substitute # ~ list of files, used to load them ...        
        self.hyperparams = hyperparams # parameters for actual data loading - various re-normalisations (between EMIT and AVIRIS domains)...
        
        self.input_products = input_products
        self.output_products = output_products
        #self.dataframe["window"] = None

        self.load_products = "all"
        if len(self.input_products) == 1 and 'mag1c' in self.input_products:
            self.load_products = "mag1c_only"

        self.weight_loss = weight_loss
        self.spatial_augmentations = spatial_augmentations
        self.window_size_sample = window_size_sample
        self.extra_products = [] if extra_products is None else extra_products
        self.add_rgb_aviris = False
        
    def add_extra_products(self, products_add: List[str]):
        p_add = [p for p in products_add if p not in self.extra_products and p not in self.input_products]
        
        self.extra_products.extend(p_add)

    def __len__(self):
        return len(self.dataframe_substitute) #.shape[0]

    def __getitem__(self, idx: int):
        data_iter = self.dataframe_substitute[idx]
        
        # use info from 'data_iter' to load the actual data
        # return as 'out_dict'
        
        if len(self.hyperparams.keys()) > 0:
            MAGIC_DIV_BY = self.hyperparams["MAGIC_DIV_BY"]
            RGB_DIV_BY = self.hyperparams["RGB_DIV_BY"]
            MAGIC_CLIP_TO = self.hyperparams["MAGIC_CLIP_TO"]
            RGB_CLIP_TO = self.hyperparams["RGB_CLIP_TO"]
            MAGIC_MULT_BY = self.hyperparams["MAGIC_MULT_BY"]
            RGB_MULT_BY = self.hyperparams["RGB_MULT_BY"]
        
        else:
            # DIV the EMIT data by
            MAGIC_DIV_BY = 240.
            RGB_DIV_BY = 20.
            # clipping too large values
            MAGIC_CLIP_TO = [0.,2.]
            RGB_CLIP_TO =   [0.,2.]
            # MULT_BY to get it back to the range we saw in the AVIRIS data ...
            MAGIC_MULT_BY = 1750.
            RGB_MULT_BY =   60.

        
        # Data:
        if self.load_products != "mag1c_only":
            rgb, magic, label, rgb_path = data_iter
        else:
            magic, label, rgb_path = data_iter
        w,h = magic.shape

        # CROP SHAPE
        w_32mult, h_32mult = math.floor(w / 32) * 32, math.floor(h / 32) * 32
        if self.load_products != "mag1c_only":
            rgb   = rgb[:, 0:w_32mult, 0:h_32mult]
        magic = magic [0:w_32mult, 0:h_32mult]
        label = label [0:w_32mult, 0:h_32mult]

        # NORMALISE
        # emit rgb has max ~22
        e_magic = np.clip(magic / MAGIC_DIV_BY, MAGIC_CLIP_TO[0], MAGIC_CLIP_TO[1]) * MAGIC_MULT_BY # project from 0-480 to 0-1750   ### < matters a bit, should be the same though kinda
        
        
        
        if self.load_products != "mag1c_only":
            e_rgb = np.clip(rgb / RGB_DIV_BY, RGB_CLIP_TO[0], RGB_CLIP_TO[1]) * RGB_MULT_BY ### < doesn't matter with mag1c baseline ...

            input_data = torch.ones((4,w_32mult, h_32mult))
            input_data[0] = torch.from_numpy(e_magic)
            input_data[1:] = torch.from_numpy(e_rgb)
        else:
            input_data = torch.ones((1,w_32mult, h_32mult))
            input_data[0] = torch.from_numpy(e_magic)
            
        has_plume = np.max(label) != 0.0
        
        plume_data = {}
        plume_data["input"] = torch.nan_to_num(input_data)
        plume_data["output"] = torch.from_numpy(label/255.).unsqueeze(0)
        plume_data["id"] = [int(idx)]
        plume_data["has_plume"] = [has_plume]
        plume_data["weight_loss"] = torch.ones_like(plume_data["output"]) # this one is faked

        plume_data["debug_rgb_path"] = [rgb_path]

        # # Note: the mask may not be needed - as long as we don't really care for the number of TNs
        # nan_mask = np.where( torch.nan_to_num(torch.from_numpy(e_rgb[0]), 999.) ==999, 1.0, 0.0)
        # plume_data["nodata_mask"] = nan_mask

        return plume_data
