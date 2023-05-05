from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from torch.utils.data import Dataset
import os
import rasterio
import torch
import rasterio.windows
import numpy as np


class STARCOPDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame,
                 input_products:List[str],
                 output_products: List[str],
                 weight_loss:Optional[str]=None,
                 spatial_augmentations=None,
                 extra_products:Optional[List[str]]=None,
                 window_size_sample:Optional[Tuple[int, int]]=None):
        self.dataframe = dataframe
        assert "folder" in self.dataframe.columns, "folder not in columns of dataframe"
        self.input_products = input_products
        self.output_products = output_products
        if "window" not in self.dataframe:
            self.dataframe["window"] = None

        self.weight_loss = weight_loss
        self.spatial_augmentations = spatial_augmentations
        self.window_size_sample = window_size_sample
        self.extra_products = [] if extra_products is None else extra_products
        self.add_rgb_aviris = False
        
    def add_extra_products(self, products_add: List[str]):
        p_add = [p for p in products_add if p not in self.extra_products and p not in self.input_products]
        
        self.extra_products.extend(p_add)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):
        data_iter = self.dataframe.iloc[idx]
        product_folder = data_iter.folder
        window = data_iter.window
        if self.window_size_sample is not None:
            # random sample window to read from window
            if window is None:
                row_off = np.random.randint(0, 512 - self.window_size_sample[0])
                col_off = np.random.randint(0, 512 - self.window_size_sample[1])

                window = rasterio.windows.Window(row_off=row_off, col_off=col_off, width=self.window_size_sample[1],
                                                 height=self.window_size_sample[0])
            else:
                row_off = window.row_off + np.random.randint(0, window.height - self.window_size_sample[0])
                col_off = window.col_off + np.random.randint(0, window.width - self.window_size_sample[1])
                window = rasterio.windows.Window(row_off=row_off, col_off=col_off, width=self.window_size_sample[1],
                                                 height=self.window_size_sample[0])


        out_dict = {}
        names_outputs = ["input", "output"]
        output_products = [self.input_products, self.output_products]
        if self.weight_loss is not None:
            names_outputs.append("weight_loss")
            output_products.append([self.weight_loss])

        for io_name, products in zip(names_outputs, output_products):
            tensors = []
            for key_name in products:
                path = os.path.join(product_folder, f"{key_name}.tif")
                with rasterio.open(path) as src:
                    tensors.append(torch.from_numpy(src.read(window=window)))
            if len(tensors) > 1:
                out_dict[io_name] = torch.cat(tensors, dim=0).float()
            elif len(tensors) == 1:
                out_dict[io_name] = tensors[0].float()

        # Add extra products to the dict. This is useful for plotting to include mag1c in WV3 models
        for key_name in self.extra_products:
            path = os.path.join(product_folder, f"{key_name}.tif")
            with rasterio.open(path) as src:
                out_dict[key_name] = torch.from_numpy(src.read(window=window))
            names_outputs.append(key_name)
        
        if self.add_rgb_aviris:
            rgb_aviris = []
            for key_name in ["TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]:
                path = os.path.join(product_folder, f"{key_name}.tif")
                with rasterio.open(path) as src:
                    rgb_aviris.append(torch.from_numpy(src.read(window=window)))
            
            out_dict["rgb_aviris"] = torch.cat(rgb_aviris, dim=0).float() / 50.

        if self.spatial_augmentations is not None:
            out_list = [out_dict[k] for k in names_outputs]
            out_list_aug = self.spatial_augmentations(*out_list)
            out_dict = {k:v for k,v in zip(names_outputs, out_list_aug)}

        # Add id and has_plume
        out_dict["id"] = str(data_iter.name)
        out_dict["has_plume"] = int(data_iter.has_plume)

        return out_dict

