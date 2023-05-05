import os

import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
from georeader import window_utils
from georeader.save_cog import save_cog
import rasterio.windows
import numpy as np
from starcop import utils
from starcop.data import aviris, mask_creation
from datetime import datetime, timezone
from tqdm import tqdm
from multiprocessing import get_context
import re


def select_non_overlapping(data: pd.DataFrame, n: int = 2, idxs: Optional[List[Any]] = None) -> List[int]:
    assert n >= 1, f"n <= 0 {n}"

    if idxs is None:
        idxs = []
    else:
        idxs = list(idxs)

    assert len(idxs) < n, f"n < len(idxs) {n} {len(idxs)}"

    for row in data.itertuples():
        if len(idxs) == 0:
            idxs.append(row.Index)
            continue

        window_idx = row.window
        if not any(rasterio.windows.intersect(data.loc[idx_other, "window"], window_idx) for idx_other in idxs):
            idxs.append(row.Index)

        if len(idxs) >= n:
            break

    return idxs


PERMIAN_BASIN_PATH = "gs://starcop/Permian/permian_2019_plume_list_with_paths.csv"
def permian_plumes_dataframe(output_size:Tuple[int,int]=(151,151)):
    """
    Loads the dataframe of the Permian basin that was preprocessed using the code of starcop.data.plume_processing.process_paths_windows

    Returns:

    """
    dataframe = pd.read_csv(PERMIAN_BASIN_PATH)
    filter_no_window = dataframe.window_col_off >= 0

    n_good = filter_no_window.sum()

    if n_good < dataframe.shape[0]:
        print(f"Found only {n_good} valid entries out of {dataframe.shape[0]} in file {PERMIAN_BASIN_PATH}")
        dataframe = dataframe[filter_no_window].reset_index(drop=True)

    dataframe["datetime"] = dataframe["name"].apply(lambda name: datetime.strptime(name, "ang%Y%m%dt%H%M%S").replace(tzinfo=timezone.utc))

    dataframe["date"] = dataframe["datetime"].apply(lambda x: datetime.strptime(x.strftime("%Y-%m-%d"), "%Y-%m-%d"))

    # Add windows as rasterio.windows.Window objects
    dataframe["window"] = dataframe.apply(
        lambda row: rasterio.windows.Window(col_off=row.window_col_off, row_off=row.window_row_off,
                                            width=row.window_width, height=row.window_height),
        axis=1)

    dataframe["id"] = dataframe.apply(
        lambda row: f"{row['name']}_r{row.window_row_off}_c{row.window_col_off}_w{row.window_width}_h{row.window_height}",
        axis=1)

    dataframe["subset"] = "train"
    for td in TEST_DATES:
        dataframe.loc[dataframe["date"] == td, "subset"] = "test"

    # Mask out banned plumes by candidate_id
    dataframe["has_plume"] = True
    dataframe = dataframe.set_index("id")

    test_samples_bad = ["ang20191018t141549_r8600_c403_w151_h151", "ang20191018t141549_r3424_c446_w151_h151",
                        "ang20191018t165503_r9641_c448_w151_h151", "ang20191018t141549_r2616_c300_w151_h151",
                        "ang20191018t153724_r13604_c135_w151_h151","ang20191018t144405_r1990_c431_w151_h151",
                        "ang20191018t144405_r1740_c34_w151_h151", "ang20191018t183859_r9089_c309_w151_h151",
                        "ang20191018t153724_r8455_c101_w151_h151", "ang20191018t165503_r9976_c226_w151_h151",
                        "ang20191018t150906_r5505_c222_w151_h151","ang20191018t172239_r4930_c291_w151_h151",
                        "ang20191018t165503_r7509_c66_w151_h151","ang20191021t160052_r9752_c418_w151_h151",
                        "ang20191018t183859_r11078_c385_w151_h151","ang20191021t173221_r8391_c86_w151_h151",
                        "ang20191018t183859_r5087_c494_w151_h151", "ang20191021t163119_r10513_c292_w151_h151",
                        "ang20191021t154726_r10577_c423_w151_h151","ang20191021t154726_r8441_c229_w151_h151",
                        "ang20191021t163119_r8462_c408_w151_h151","ang20191021t174954_r8627_c460_w151_h151",
                        "ang20191021t153008_r8754_c366_w151_h151", "ang20191021t154726_r7273_c203_w151_h151",
                        "ang20191021t183204_r3408_c492_w151_h151", "ang20191018t174629_r13283_c433_w151_h151",
                        "ang20191021t154726_r10547_c373_w151_h151","ang20191021t154726_r8361_c253_w151_h151",
                        "ang20191021t154726_r10684_c481_w151_h151","ang20191021t160052_r7282_c221_w151_h151",
                        "ang20191021t154726_r10825_c8_w151_h151","ang20191021t153008_r5387_c384_w151_h151"]

    dataframe = dataframe.drop(test_samples_bad)

    if output_size != (151, 151):
        dataframe["window"] = dataframe.window.apply(lambda wv: window_utils.pad_window_to_size(wv, output_size))

    return dataframe


PERMIAN_MAG1C_STATS_DATAFRAME = "gs://starcop/Permian/stats_mag1c.csv"
TEST_DATES = ["2019-10-25", "2019-10-21", "2019-10-18"]


def permian_mag1c_stats_dataframe(plumes_dataframe:pd.DataFrame):
    mag1c_stats = pd.read_csv(PERMIAN_MAG1C_STATS_DATAFRAME)
    mag1c_stats = mag1c_stats[mag1c_stats["window_col_off"] >= 0]
    mag1c_stats["folder"] = mag1c_stats["folder"].apply(lambda x: x if x.endswith("/") else x + "/")
    mag1c_stats["name"] = mag1c_stats["folder"].apply(lambda x: x.split("/")[-2])
    mag1c_stats["datetime"] = mag1c_stats["name"].apply(
        lambda name: datetime.strptime(name, "ang%Y%m%dt%H%M%S").replace(tzinfo=timezone.utc))

    mag1c_stats["date"] = mag1c_stats["datetime"].apply(lambda x: datetime.strptime(x.strftime("%Y-%m-%d"), "%Y-%m-%d"))

    # Set index
    mag1c_stats["id"] = mag1c_stats.apply(
        lambda
            row: f"{row['name']}_r{row.window_row_off}_c{row.window_col_off}_w{row.window_width}_h{row.window_height}",
        axis=1)
    mag1c_stats = mag1c_stats.set_index("id")

    mag1c_stats["percentage_valids"] = mag1c_stats["count"] / (
                mag1c_stats["window_width"] * mag1c_stats["window_height"])

    mag1c_stats["has_plume"] = False

    # Set here plumes that are not labeled
    mag1c_stats.loc["ang20191018t183859_r2304_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20191018t183859_r2560_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20191021t190136_r4096_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20191018t141549_r2560_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190926t172904_r512_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190926t184029_r6144_c256_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190927t164322_r3328_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190923t185208_r4608_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190926t172904_r768_c0_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190926t184029_r6400_c256_w512_h512", "has_plume"] = True
    mag1c_stats.loc["ang20190927t153023_r8192_c0_w512_h512", "has_plume"] = True # Big One
    mag1c_stats.loc["ang20191005t215301_r5120_c0_w512_h512", "has_plume"] = True  # Big One
    # ang20190927t184620_r6144_c0_w512_h512 maybe?
    mag1c_stats.loc["ang20191007t195115_r768_c0_w512_h512", "has_plume"] = True  # Big One
    mag1c_stats.loc["ang20191012t162223_r3072_c0_w512_h512", "has_plume"] = True  # Big One
    mag1c_stats.loc["ang20191005t215301_r4864_c0_w512_h512", "has_plume"] = True  # Big One

    mag1c_stats["window"] = mag1c_stats.apply(
        lambda row: rasterio.windows.Window(col_off=row.window_col_off, row_off=row.window_row_off,
                                            width=row.window_width, height=row.window_height),
        axis=1)

    # Checks if the window has a plume
    def has_plume(row:pd.Series) -> bool:
        if row.has_plume:
            return True

        window_plumes_file = plumes_dataframe[plumes_dataframe.folder == row.folder].window
        window_plumes_not_labeled = mag1c_stats[(mag1c_stats.folder == row.folder) & mag1c_stats.has_plume].window
        if window_plumes_not_labeled.shape[0] > 0:
            window_plumes_file = pd.concat([window_plumes_file, window_plumes_not_labeled])

        return window_plumes_file.apply(lambda x: rasterio.windows.intersect(x, row.window)).any()

    mag1c_stats["has_plume"] = mag1c_stats.apply(has_plume, axis=1)

    # Set flight lines for training and testing
    mag1c_stats["subset"] = "train"
    for td in TEST_DATES:
        mag1c_stats.loc[mag1c_stats["date"] == td, "subset"] = "test"

    # ang20191013t152730_r13568_c0_w512_h512

    # ang20191011t152413_r4608_c0_w512_h512 maybe
    return mag1c_stats


class WindowDataset:
    def __init__(self, dataframe:pd.DataFrame, products:List[str],
                 read_label_path:bool = False, read_rgb_path:bool = False,
                 wavelengths:Optional[List[float]]=None,
                 as_geotensor:bool=True,
                 output_size:Optional[Tuple[int, int]]=None,
                 normalize_by_acquisition_date:bool=True, proposed_mask:bool=True):
        """

        Args:
            dataframe:
            products:
            read_label_path:
            read_rgb_path:
            wavelengths: List of wavelengths in nm to read form the AVIRIS file (will read the closest band to the one
                given)
            as_geotensor:
            output_size: images will be padded to this size
            normalize_by_acquisition_date: normalize radiances by acquisition date
        """
        self.dataframe = dataframe
        self.products = products
        self.as_geotensor = as_geotensor
        self.read_label_path = read_label_path
        self.read_rgb_path = read_rgb_path
        self.proposed_mask = proposed_mask

        # Cache readers to avoid re-constructing the object
        self.product_readers:Dict[str, RasterioReader] = {}

        # Conversion to TOA AVIRIS bands and S2A/S2B and WV3 rasters
        self.normalize_by_acquisition_date = normalize_by_acquisition_date
        self.toa_correction_factor:Dict[str, float] = {}

        # wavelengths_products is used to cache the wavelengths of the products
        self.wavelengths_products: Dict[str, np.array] = {}
        if wavelengths is not None:
            self.wavelengths = np.array(wavelengths)
            self.wavelengths_names = [f"{w:.0f}nm" for w in self.wavelengths]
        else:
            self.wavelengths = None
            self.wavelengths_names = None

        # Force reads with a fixed size
        self.output_size = output_size
        if self.read_label_path or self.read_rgb_path:
            self.dataframe["window_labels"] = self.dataframe.window.apply(
                lambda wv: rasterio.windows.Window(row_off=0, col_off=0, width=wv.width,
                                                   height=wv.height))
        if output_size is not None:
            self.dataframe = self.dataframe.copy()

            # update windows
            self.dataframe["window"] = self.dataframe.window.apply(lambda wv: window_utils.pad_window_to_size(wv, self.output_size))
            self.dataframe["window_labels"] = self.dataframe.window_labels.apply(lambda wv: window_utils.pad_window_to_size(wv, self.output_size))

    def __len__(self):
        return  self.dataframe.shape[0]

    def __getitem__(self, idx:int):
        data_iter = self.dataframe.iloc[idx]
        product_folder = data_iter.folder
        window = data_iter.window

        # Select the products to read according to the wavelength
        products_read_extra = []
        wavelengths_names = []
        if self.wavelengths is not None:
            wavelengths_names = self.wavelengths_names
            if product_folder not in self.wavelengths_products:
                path_json = os.path.join(product_folder, "metadata.json")
                metadata = utils.read_json_from_gcp(path_json)
                self.wavelengths_products[product_folder] = np.array(metadata["wavelengths"])

            wavelengths_product = self.wavelengths_products[product_folder]
            products_read_extra = [f"{a}" for a in np.argmin(np.abs(np.array(self.wavelengths)[:, np.newaxis] - wavelengths_product), axis=1)]

        out_dict = {}
        for path_name, key_name in zip(self.products + products_read_extra, self.products + wavelengths_names):
            path = os.path.join(product_folder, f"{path_name}.tif")
            if path not in self.product_readers:
                self.product_readers[path] = RasterioReader(path)

            product_reader = self.product_readers[path]
            out_dict[key_name] = product_reader.read_from_window(window=window,boundless=True).load(boundless=True)

            # Set fill value to zero (works for radiances and mag1c)
            nodata_values = out_dict[key_name].values == out_dict[key_name].fill_value_default
            out_dict[key_name].values[nodata_values] = 0
            out_dict[key_name].fill_value_default = 0

            # conversion to toa
            if self.normalize_by_acquisition_date and (path_name.startswith("S2") or path_name.startswith("WV") or path_name.isnumeric()):
                if product_folder not in self.toa_correction_factor:
                    center_coords = product_reader.transform * (product_reader.shape[-1] // 2, product_reader.shape[-2] // 2)
                    self.toa_correction_factor[product_folder] = aviris.observation_date_correction_factor(center_coords, data_iter.datetime.to_pydatetime(),
                                                                                                       crs_coords=product_reader.crs)
                toa_correction_factor = self.toa_correction_factor[product_folder]
                if path_name.startswith("S2") or path_name.startswith("WV"):
                    sensor, band = path_name.split("_")
                    if len(band) == 2:
                        band = f"B0{band[-1]}"
                    solar_irradiance = aviris.SOLAR_IRRADIANCE[sensor][band]
                    out_dict[key_name].values *=  (toa_correction_factor / 100  / solar_irradiance)
                    # Clip TOA reflectances to 0-2
                    out_dict[key_name].values = np.clip(out_dict[key_name].values, 0, 2)
                else:
                    out_dict[key_name].values *= toa_correction_factor

            # Clamp mag1c values
            if path_name == "mag1c":
                out_dict[key_name].values = np.clip(out_dict[key_name].values, 0, 10_000)

            if not self.as_geotensor:
                out_dict[key_name] = out_dict[key_name].values

        if self.proposed_mask:
            binary_mask = mask_creation.proposed_mask(out_dict["label_rgba"].values, out_dict["mag1c"].values).astype(np.uint8)
            out_dict["labelbinary"] = GeoTensor(binary_mask, transform=out_dict["mag1c"].transform, crs=out_dict["mag1c"].crs,
                                                fill_value_default=None)
            if not self.as_geotensor:
                out_dict["labelbinary"] = out_dict["labelbinary"].values

        # Read labels [DEPRECATED, now reads from the join product ["label_rgba.tif"]
        if self.read_label_path:
            window_labels =  data_iter.window_labels
            if data_iter.label_path not in self.product_readers:
                self.product_readers[data_iter.label_path] = RasterioReader(data_iter.label_path,
                                                                            window_focus=window_labels)

            product_reader = self.product_readers[data_iter.label_path]
            out_dict["label"] = product_reader.load(boundless=True)

            if not self.as_geotensor:
                out_dict["label"] = out_dict["label"].values


        if self.read_rgb_path:
            window_labels = data_iter.window_labels
            if data_iter.rgb_path not in self.product_readers:
                self.product_readers[data_iter.rgb_path] = RasterioReader(data_iter.rgb_path,
                                                                          window_focus=window_labels)

            product_reader = self.product_readers[data_iter.rgb_path]
            out_dict["rgb"] = product_reader.load(boundless=True)
            if not self.as_geotensor:
                out_dict["rgb"] = out_dict["rgb"].values

        return out_dict

    def cache_item(self, idx, output_path, overwrite):
        fs = utils.get_filesystem(output_path)
        folder_idx_path = os.path.join(output_path, self.dataframe.index[idx])
        make_dir_if_needed(folder_idx_path, fs)

        plume_data = self[idx]
        for k in plume_data:
            if self.normalize_by_acquisition_date and (
                    k.startswith("S2") or k.startswith("WV") or k.endswith("nm") or k.isnumeric()):
                if k.endswith("nm") or k.isnumeric():
                    # This really it's not TOA is "normalized radiance" because we don't divide by the solar irradiance
                    k_save = f"TOA_AVIRIS_{k}"
                else:
                    k_save = f"TOA_{k}"
            else:
                k_save = k

            path_save = os.path.join(folder_idx_path, f"{k_save}.tif")
            if overwrite or not fs.exists(path_save):
                v = plume_data[k]
                if k == "label_rgba":
                    descriptions = ["r", "g", "b", "a"]
                else:
                    descriptions = [k_save, ]
                save_cog(v, path_save, descriptions=descriptions, profile={"BLOCKSIZE": 128})

    def cache(self, output_path:str, dataframe_name:str, overwrite:bool=False, num_processes:int=1):
        fs = utils.get_filesystem(output_path)

        assert self.as_geotensor, f"In order to cache, files must be saved as geotensor"
        make_dir_if_needed(output_path, fs)

        if num_processes > 1:
            with get_context("spawn").Pool(num_processes) as p:
                r = tqdm(p.imap(cache_map, [(self, idx, output_path, overwrite)
                                            for idx in range(len(self))], chunksize=10), total=len(self))
        else:
            for idx in tqdm(range(len(self)), total=len(self), desc="Caching data"):
                self.cache_item(idx, output_path, overwrite)

        csv_original_path = os.path.join(output_path, f"{dataframe_name}_sampled_data.csv")
        columns_copy = [c for c in self.dataframe.columns if c != "window"]
        if overwrite or not fs.exists(csv_original_path):
            self.dataframe[columns_copy].to_csv(csv_original_path, index=True)

        dataframe_new = self.dataframe.copy()
        dataframe_new["folder"] = [os.path.join(output_path, self.dataframe.index[idx]) for idx in range(len(self))]
        dataframe_new["window_col_off"] = 0
        dataframe_new["window_row_off"] = 0
        dataframe_new["window_width"] = self.output_size[-1]
        dataframe_new["window_height"] = self.output_size[-2]

        csv_path = os.path.join(output_path, f"{dataframe_name}.csv")
        if overwrite or not fs.exists(csv_path):
            dataframe_new[columns_copy].to_csv( csv_path, index=True)

def cache_map(tuple_arg):
    WindowDataset.cache_item(*tuple_arg)
    return True

def make_dir_if_needed(path, fs):
    if not path.startswith("gs://"):
        fs.makedirs(path, exist_ok=True)

WV3_BANDS = [f"WV3_SWIR{w+1}"for w in range(8)]
S2_BANDS = ["B1", "B2",
            "B3", "B4",
            "B5", "B6",
            "B7", "B8",
            "B8A", "B9",
            "B10", "B11", "B12"]

S2A_BANDS = [f"S2A_{b}" for b in S2_BANDS]
S2B_BANDS = [f"S2B_{b}" for b in S2_BANDS]


def sampling_no_plumes(no_plumes:pd.DataFrame, n_hard:int, n_random:int,percentage_valids:float=.8, seed:int=42) -> pd.DataFrame:
    np.random.seed(seed)
    files = no_plumes["name"].unique()
    random_hard_column = []
    indexes_test = []
    for f in files:
        plumes_f = no_plumes[no_plumes["name"] == f]

        # keep plumes with more than 90% valids
        plumes_f = plumes_f[plumes_f["percentage_valids"] >= percentage_valids].sort_values(by="mean", ascending=False)
        idx_hard = select_non_overlapping(plumes_f, n=n_hard)
        random_hard_column_iter = ["hard" for _ in range(len(idx_hard))]

        plumes_permuted = plumes_f.iloc[np.random.permutation(plumes_f.shape[0])]
        idx_random_and_hard = select_non_overlapping(plumes_permuted, n=n_hard + n_random,
                                                     idxs=idx_hard)

        random_hard_column_iter += ["random" for _ in range(len(idx_random_and_hard) - len(idx_hard))]

        random_hard_column.extend(random_hard_column_iter)
        indexes_test.extend(idx_random_and_hard)

        # Sample 4 non-overlapping windows
        # 2 with the biggest amount of confounders

    no_plumes_selected = no_plumes.loc[indexes_test].copy()
    no_plumes_selected["difficulty"] = random_hard_column
    no_plumes_selected["qplume"] = 0
    no_plumes_selected["candidate_id"] = ""
    no_plumes_selected["label_path"] = ""

    return no_plumes_selected


def _cache_data_permian_2019(products:List[str], subset:str, folder:str, overwrite:bool=False, num_processes:int=1):

    dt = pd.read_csv(f"gs://starcop/Permian/selected_{subset}.csv")
    dt["window"] = dt.apply(
        lambda row: rasterio.windows.Window(col_off=row.window_col_off, row_off=row.window_row_off,
                                            width=row.window_width, height=row.window_height),
        axis=1)
    dt = dt.set_index("id")
    dt["datetime"] = dt["name"].apply(
        lambda name: datetime.strptime(name, "ang%Y%m%dt%H%M%S").replace(tzinfo=timezone.utc))

    products_real = set([p for p in products if (p != "labelbinary")])
    compute_proposed_mask = "labelbinary" in products
    if compute_proposed_mask:
        products_real = products_real.union({"mag1c", "label_rgba"})
        if overwrite:
            compute_proposed_mask = True
        else:
            compute_proposed_mask = not dataframe["name"].apply(lambda x: os.path.exists(os.path.join(folder,x, "labelbinary.tif"))).all()

    products_generate = []
    for p in products_real:
        if overwrite:
            exists_all = False
        else:
            exists_all = dataframe["name"].apply(lambda x: os.path.exists(os.path.join(folder,x, f"{p}.tif"))).all()

        if exists_all:
            print(f"Product {p} will NOT be generated")
        else:
            products_generate.append(p)
            print(f"Product {p} will be generated")


    # Split products generate in products and wavelengths array
    products_generate_names = []
    products_generate_wavelengths = []
    for p in products_generate:
        if p.startswith("TOA_AVIRIS"):
            match = re.match(r"TOA_AVIRIS_(\d+)nm", p)
            assert match is not None, f"Product name {p} does not match the expected pattern: TOA_AVIRIS_(\d+)nm"
            wavelength = int(match.groups()[0])
            products_generate_wavelengths.append(wavelength)
        else:
            products_generate_names.append(p)

    dataset = WindowDataset(dataframe=dt,
                            products= products_generate_names,
                            wavelengths=products_generate_wavelengths,
                            # products=["mag1c", "label_rgba"] + WV3_BANDS + S2A_BANDS + S2B_BANDS,
                            # wavelengths=[640, 550, 460, 2004, 2109, 2310, 2350, 2360],
                            read_rgb_path=False, read_label_path=False,
                            proposed_mask=compute_proposed_mask,
                            normalize_by_acquisition_date=True,
                            as_geotensor=True,
                            output_size=(512, 512))

    # TODO Update labeled data

    dataset.cache(folder, subset, overwrite=overwrite, num_processes=num_processes)

def generate_train_data_permian_2019(folder:str, products:List[str], overwrite:bool=False, num_processes:int=1):
    # The file selecting the training data is generated in select_train_data_aviris_permian_2019.ipynb
    return _cache_data_permian_2019(products, subset="train", folder=folder, overwrite=overwrite, num_processes=num_processes)


def generate_test_data_permian_2019(folder:str, products:List[str], overwrite:bool=False, num_processes:int=1):
    # The file selecting the testing data is generated in select_test_data_aviris_permian_2019.ipynb
    return _cache_data_permian_2019(products, subset="test", folder=folder, overwrite=overwrite, num_processes=num_processes)



if __name__ == "__main__":
    import argparse
    from time import time

    parser = argparse.ArgumentParser(usage="Generate Permian 2019 training and test datasets")
    parser.add_argument('--root_folder', type=str, required=True,
                        help=f'Root folder to cache the data')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)
    parser.add_argument('--clean_root_folder', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_processes', type=int, required=False,default=1,
                        help=f'Number of process to read cache the data')

    args = parser.parse_args()
    start = time()
    generate_train_data_permian_2019(args.root_folder, overwrite=args.overwrite, num_processes=args.num_processes)
    print(f"Took {(time()-start)/60:.2f} minutes to produce the training set")
    start = time()
    generate_test_data_permian_2019(args.root_folder, overwrite=args.overwrite, num_processes=args.num_processes)
    print(f"Took {(time() - start) / 60:.2f} minutes to produce the test set")

    if args.clean_root_folder:
        dataframe_train = pd.read_csv(os.path.join(args.root_folder, "train.csv"))
        dataframe_test = pd.read_csv(os.path.join(args.root_folder, "test.csv"))
        dataframe = pd.concat([dataframe_test, dataframe_train])
        dataframe = dataframe.set_index("id")

        from glob import glob
        from shutil import rmtree
        # TODO use fsspec?

        folders = glob(os.path.join(args.root_folder, "ang*"))
        print(f" Found {len(folders)} folders. Generated files {dataframe.shape[0]}")
        count_exist = 0
        count_doesnt_exist = 0
        for folder in folders:
            folder_name = os.path.basename(folder)
            if folder_name in dataframe.index:
                count_exist += 1
            else:
                rmtree(folder)
                count_doesnt_exist += 1

        print(f"Exists: {count_exist} Doesn't exist {count_doesnt_exist}")






