# Purpose:
# we want to work with the emit data in the same way as we did with the aviris dataset and dataloaders.
# Ideally this should allow using the same functions. We mostly care about validation, not necessarily the training part...
# needs to do only:
# - data_module.prepare_data()
# - dataloader = data_module.test_dataloader()


import os
import pytorch_lightning as pl
from typing import Optional, Tuple, List

import kornia.augmentation as K
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import rasterio.windows
import numpy as np
import torch
from tqdm import tqdm
import logging


from starcop.emit_tools.emit_dataset import STARCOPEMITDataset



class EMITDataModule(pl.LightningDataModule):
    def __init__(self, settings, labels_filename = "label.tif", hyperparams = {}, root_folder = "/media/vitek/4E3EC8833EC86595/Vitek/Datasets/EMIT DATA/PermianBasinEMIT_sample/vitek_emit_data/EMIT_DATA_magic_outputs/EMIT_DATASET_v0b"):
        # labels_filename = "label.tif" or "label_released.tif" 
        super().__init__()
        self.settings = settings
        self.products_plot = settings.products_plot

        self.batch_size = self.settings.dataloader.batch_size
        self.num_workers = self.settings.dataloader.num_workers
        self.input_products = self.settings.dataset.input_products
        self.output_products = self.settings.dataset.output_products
        
        self.load_products = "all"
        if len(self.input_products) == 1 and 'mag1c' in self.input_products:
            self.load_products = "mag1c_only"
        print("Made the EMITDataModule with these self.input_products", self.input_products, "will load these:", self.load_products)
        
        self.labels_filename = labels_filename
        self.hyperparams = hyperparams
        self.root_folder = root_folder
        
        if self.settings.dataset.use_weight_loss:
            self.weight_loss = self.settings.dataset.weight_loss
        else:
            self.weight_loss = None

        self.weight_sampling = self.settings.dataset.weight_sampling

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.
        This method is called once per GPU per run.
        Args:
            stage: stage to set up
        """    

    def prepare_data(self):
        """
        Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        log = logging.getLogger(__name__)
        if self.weight_loss is not None:
            extra_types = ["input"]
            weight_loss_list = [self.weight_loss]
        else:
            extra_types = []
            weight_loss_list = []
        
        if self.settings.model.model_mode == "segmentation_output":
            model_output_type = "mask"
        else:
            model_output_type = "input"


        # emit data
        from starcop.emit_tools.emit_data_utils import load_emit_dataset, load_data
        dataset_paths = load_emit_dataset(self.root_folder, labels_name=self.labels_filename)
        emit_data = load_data(dataset_paths, self.load_products)
        
        self.test_dataset = STARCOPEMITDataset(emit_data,
                                                   input_products=self.input_products,
                                                   weight_loss=self.weight_loss,
                                                   output_products=self.output_products,
                                                   hyperparams = self.hyperparams
                                              )
        
        self.test_dataset_plot = STARCOPEMITDataset(emit_data,
                                                        input_products=self.input_products,
                                                        weight_loss=self.weight_loss,
                                                        output_products=self.output_products,
                                                        hyperparams = self.hyperparams
                                                   )
        
        if "rgb_aviris" in self.products_plot and not all(b in self.input_products for b in ["TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]):
            self.test_dataset_plot.add_rgb_aviris = True
        if "mag1c" in self.products_plot and "mag1c" not in self.input_products:
            self.test_dataset_plot.add_extra_products(["mag1c"])

        log.info("Data module ready")
        log.info(f"Input products: {self.input_products} Output products: {self.output_products} Weight loss: {self.weight_loss}")
        log.info(f"Test dataset {len(self.test_dataset)}")
    
    def test_dataloader(self, num_workers:Optional[int]=None, batch_size:Optional[int]=None):
        """Initializes and returns the test dataloader"""
        num_workers = num_workers or self.num_workers
        batch_size = batch_size or self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

