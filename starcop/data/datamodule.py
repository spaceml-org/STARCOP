import os
import pytorch_lightning as pl
from typing import Optional, Tuple, List

import kornia.augmentation as K
from torch.utils.data import DataLoader, WeightedRandomSampler
from starcop.data import dataset
import pandas as pd
from . import feature_extration
import rasterio.windows
import numpy as np
import torch
from tqdm import tqdm
import logging


def tiled_dataframe(dataframe:pd.DataFrame, tile_size:Tuple[int, int],
                    overlap:Tuple[int, int], output_products:List[str],
                    num_workers:int=2) -> pd.DataFrame:
    from georeader.slices import create_windows
    dataframe_tiled_list = []
    for row in dataframe.reset_index().to_dict(orient="records"):
        del row["window_row_off"]
        del row["window_col_off"]
        del row["window_width"]
        del row["window_height"]
        for w in create_windows((512, 512),
                                window_size=tile_size, overlap=overlap, include_incomplete=False):
            row_copy = dict(row)
            row_copy["window"] = w
            dataframe_tiled_list.append(row_copy)

    dataframe_tiled = pd.DataFrame(dataframe_tiled_list)

    # Compute weighting score
    dataset_labels = dataset.STARCOPDataset(dataframe_tiled,
                                            input_products=[],
                                            output_products=output_products,
                                            weight_loss=None,
                                            spatial_augmentations=None,
                                            window_size_sample=None)
    dl_labels = DataLoader(dataset_labels, batch_size=1, num_workers=num_workers, shuffle=False)

    frac_positives = []
    for label in tqdm(dl_labels, total=len(dl_labels), desc="Computing label statistics"):
        with torch.no_grad():
            frac_positives.append(torch.sum(label["output"]).item() / np.prod(tuple(label["output"].shape)))

    dataframe_tiled["frac_positives"] = np.array(frac_positives)
    dataframe_tiled["has_plume"] = dataframe_tiled["frac_positives"] > (10 / 64**2)

    for attr_name in ["col_off", "row_off", "width", "height"]:
        dataframe_tiled[f"window_{attr_name}"] = dataframe_tiled["window"].apply(
            lambda x: None if x is None else getattr(x, attr_name))
    
    dataframe_tiled["id_original"] = dataframe_tiled["id"].copy()
    dataframe_tiled["id"] = dataframe_tiled.apply(
        lambda
            row: f"{row['id']}_r{row.window_row_off}_c{row.window_col_off}_w{row.window_width}_h{row.window_height}",
        axis=1)
    
    dataframe_tiled = dataframe_tiled.set_index("id")

    return dataframe_tiled



class Permian2019DataModule(pl.LightningDataModule):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.products_plot = settings.products_plot

        self.batch_size = self.settings.dataloader.batch_size
        self.num_workers = self.settings.dataloader.num_workers
        self.input_products = self.settings.dataset.input_products
        self.output_products = self.settings.dataset.output_products
        self.training_size = self.settings.dataset.training_size
        self.training_size_overlap = self.settings.dataset.training_size_overlap
        self.root_folder = self.settings.dataset.root_folder
        self.train_csv = self.settings.dataset.train_csv
        self.test_csv = "test.csv"
        
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

    def load_dataframe(self, path) -> pd.DataFrame:
        train_dataframe = pd.read_csv(path)
        train_dataframe["window"] = train_dataframe.apply(
            lambda row: rasterio.windows.Window(col_off=row.window_col_off, row_off=row.window_row_off,
                                                width=row.window_width, height=row.window_height),
            axis=1)
        train_dataframe["folder"] = train_dataframe["id"].apply(lambda x: os.path.join(self.root_folder, x))
        train_dataframe = train_dataframe.set_index("id")
        return train_dataframe


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

        self.train_augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            keepdim=True,
            data_keys=["input" , model_output_type] + extra_types,
        )

        # Feature extraction if needed
        raw_bands_available = feature_extration.raw_bands_available()
        self.features_extract = [f for f in self.input_products + self.output_products + weight_loss_list if
                                 f not in raw_bands_available]

        # Which products are needed as input
        self.raw_bands = [f for f in self.input_products + self.output_products + weight_loss_list if
                          f in raw_bands_available]

        train_dataset_path = os.path.join(self.root_folder, self.train_csv)
        test_dataset_path = os.path.join(self.root_folder, self.test_csv)


        products = list(self.raw_bands)
        for f in self.features_extract:
            products.extend(feature_extration.FEATURES[f]["inputs"])

        # Download the data to self.root_folder if needed
        if not os.path.exists(train_dataset_path):
            log.info(f"File for training dataset {train_dataset_path} not found we will download the data to {self.root_folder}")
            from starcop.data import sampling_dataset
            sampling_dataset.generate_train_data_permian_2019(self.root_folder, num_processes=self.num_workers,
                                                              products=products)

        # Download the data to self.root_folder if needed
        if not os.path.exists(test_dataset_path):
            from starcop.data import sampling_dataset
            log.info(
                f"File for testing dataset {test_dataset_path} not found we will download the data to {self.root_folder}")
            sampling_dataset.generate_test_data_permian_2019(self.root_folder, num_processes=self.num_workers,
                                                             products=products)

        # Process train dataframe
        self.train_dataframe_original = self.load_dataframe(train_dataset_path)

        # Extract features if needed
        if len(self.features_extract) > 0:
            feature_extration.extract_features(self.features_extract, self.train_dataframe_original)

        # slice train_dataframe in windows
        if np.any(np.array(self.training_size) < np.array([512, 512])):
            name_csv, ext = os.path.splitext(self.train_csv)
            train_dataset_path_tiled = os.path.join(self.root_folder, f"{name_csv}_tiled_{self.training_size[0]}_{self.training_size[1]}{ext}")
            if not os.path.exists(train_dataset_path_tiled):
                log.info(f"Tiled dataset {train_dataset_path_tiled} not found. Generating")
                train_dataframe = tiled_dataframe(self.train_dataframe_original, tile_size=self.training_size, overlap=self.training_size_overlap,
                                                  output_products=self.output_products, num_workers=self.num_workers)
                train_dataframe[[c for c in train_dataframe.columns if c != "window"]].to_csv(train_dataset_path_tiled)
            else:
                log.info(f"Loading tiled dataset {train_dataset_path_tiled}")
                train_dataframe = pd.read_csv(train_dataset_path_tiled)
                train_dataframe["window"] = train_dataframe.apply(
                    lambda row: rasterio.windows.Window(col_off=row.window_col_off, row_off=row.window_row_off,
                                                        width=row.window_width, height=row.window_height),
                    axis=1)
                train_dataframe["folder"] = train_dataframe["id_original"].apply(lambda x: os.path.join(self.root_folder, x))
                train_dataframe = train_dataframe.set_index("id")
        else:
            train_dataframe = self.train_dataframe_original
        

        self.train_dataset = dataset.STARCOPDataset(train_dataframe,
                                                    input_products=self.input_products,
                                                    output_products=self.output_products,
                                                    weight_loss=self.weight_loss,
                                                    spatial_augmentations=self.train_augmentations,
                                                    window_size_sample=None)
        
        self.train_dataset_plot = dataset.STARCOPDataset(train_dataframe,
                                                         input_products=self.input_products,
                                                         output_products=self.output_products,
                                                         weight_loss=self.weight_loss,
                                                         spatial_augmentations=None,
                                                         window_size_sample=None)

        self.train_dataset_non_tiled = dataset.STARCOPDataset(self.train_dataframe_original,
                                                              input_products=self.input_products,
                                                              output_products=self.output_products,
                                                              weight_loss=self.weight_loss,
                                                              spatial_augmentations=None,
                                                              window_size_sample=None)
        

        # Process test dataframe
        test_dataframe = self.load_dataframe(test_dataset_path)
        test_dataframe = test_dataframe.sort_values(["has_plume","qplume"],ascending=False)

        if len(self.features_extract) > 0:
            feature_extration.extract_features(self.features_extract, test_dataframe)

        self.test_dataset = dataset.STARCOPDataset(test_dataframe,
                                                   input_products=self.input_products,
                                                   weight_loss=self.weight_loss,
                                                   output_products=self.output_products)
        
        self.test_dataset_plot = dataset.STARCOPDataset(test_dataframe,
                                                        input_products=self.input_products,
                                                        weight_loss=self.weight_loss,
                                                        output_products=self.output_products)
        
        if "rgb_aviris" in self.products_plot and not all(b in self.input_products for b in ["TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]):
            self.train_dataset_plot.add_rgb_aviris = True
            self.test_dataset_plot.add_rgb_aviris = True
            
        
        if "mag1c" in self.products_plot and "mag1c" not in self.input_products:
            self.train_dataset_plot.add_extra_products(["mag1c"])
            self.test_dataset_plot.add_extra_products(["mag1c"])

        self.val_dataset = self.test_dataset
        log.info("Data module ready")
        log.info(f"Input products: {self.input_products} Output products: {self.output_products} Weight loss: {self.weight_loss}")
        log.info(f"Train dataset {len(self.train_dataset)} chipsize: {self.training_size}")
        log.info(f"Val dataset {len(self.val_dataset)}")
        log.info(f"Test dataset {len(self.test_dataset)}")
    
    def train_plot_dataloader(self, batch_size:int,num_workers:int=0):
        if self.weight_sampling:
            # Set weight per sample
            train_dataframe = add_sample_weight(self.train_dataset_plot.dataframe)

            weight_random_sampler = WeightedRandomSampler(train_dataframe["sample_weight"].values,
                                                          num_samples=len(self.train_dataset_plot),
                                                          replacement=True) # Must be true otherwise we should lower num_samples
            shuffle = False
        else:
            weight_random_sampler = None
            shuffle = True
        
        return DataLoader(self.train_dataset_plot, batch_size=batch_size,
                          num_workers=num_workers, sampler=weight_random_sampler,
                          shuffle=shuffle)
    
    def test_plot_dataloader(self, batch_size:int,num_workers:int=0):
        return DataLoader(self.test_dataset_plot, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

    def train_dataloader(self, num_workers:Optional[int]=None, batch_size:Optional[int]=None):
        """Initializes and returns the training dataloader"""
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers

        if self.weight_sampling:
            # Set weight per sample
            train_dataframe = add_sample_weight(self.train_dataset.dataframe)

            weight_random_sampler = WeightedRandomSampler(train_dataframe["sample_weight"].values,
                                                          num_samples=len(self.train_dataset),
                                                          replacement=True) # Must be true otherwise we should lower num_samples
            shuffle=False
        else:
            weight_random_sampler = None
            shuffle=True
        
        return DataLoader(self.train_dataset, batch_size=batch_size,
                          num_workers=num_workers, sampler=weight_random_sampler,
                          shuffle=shuffle)

    def val_dataloader(self, num_workers:Optional[int]=None, batch_size:Optional[int]=None):
        """Initializes and returns the validation dataloader"""
        num_workers = num_workers or self.num_workers
        batch_size = batch_size or self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    def test_dataloader(self, num_workers:Optional[int]=None, batch_size:Optional[int]=None):
        """Initializes and returns the test dataloader"""
        num_workers = num_workers or self.num_workers
        batch_size = batch_size or self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)


def add_sample_weight(dataframe:pd.DataFrame) -> pd.DataFrame:
    plume_fraction = np.sum(dataframe["has_plume"]) / dataframe.shape[0]
    plume_weight = 1 / plume_fraction
    non_plume_weight = 1 / (1 - plume_fraction)
    dataframe["sample_weight"] = dataframe["has_plume"].apply(
        lambda x: plume_weight if x else non_plume_weight)
    return dataframe


# easy_train_dataset = data_module.train_dataset.dataframe[(data_module.train_dataset.dataframe.qplume >= 1000) | ~data_module.train_dataset.dataframe.has_plume].copy()
# frac_has_plume = easy_train_dataset.has_plume.sum() / easy_train_dataset.shape[0] + .03
# easy_train_dataset_select = easy_train_dataset.has_plume | (np.random.rand(easy_train_dataset.shape[0]) <= frac_has_plume)
# easy_train_dataset = easy_train_dataset[easy_train_dataset_select].copy()
# easy_train_dataset.groupby("has_plume")[["name"]].count()
# easy_train_dataset[[c for c in easy_train_dataset.columns if c != "window"]].to_csv("/AVIRISNG/Permian2019/train_easy.csv")