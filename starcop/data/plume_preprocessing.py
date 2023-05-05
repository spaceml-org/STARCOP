from georeader.abstract_reader import GeoData
import rasterio.windows
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

def _is_exact_round(x, precision=1e-6):
    return abs(round(x)-x) < precision

def window_from_product(data_in: GeoData, data_other: GeoData) -> rasterio.windows.Window:
    assert data_in.crs == data_other.crs

    transform_data_other = data_other.transform
    transform_data_in = data_in.transform

    assert (transform_data_other.a == transform_data_in.a) and (transform_data_other.b == transform_data_in.b) and (
                transform_data_other.d == transform_data_in.d) and (transform_data_other.e == transform_data_in.e)

    coords_upper_left = transform_data_other.c, transform_data_other.f

    pixel_upper_left = ~transform_data_in * coords_upper_left
    if _is_exact_round(pixel_upper_left[0]) and _is_exact_round(pixel_upper_left[1]):
        pixel_upper_left[0] = int(round(pixel_upper_left[0]))
        pixel_upper_left[1] = int(round(pixel_upper_left[1]))

    return rasterio.windows.Window(row_off=pixel_upper_left[1], col_off=pixel_upper_left[0],
                                   width=data_other.shape[-2], height=data_other.shape[-1])


def process_paths_windows(name_file: str = "gs://starcop/Permian/permian_2019_plume_list.csv",
                          filepath_out:str = "gs://starcop/Permian/permian_2019_plume_list_with_paths.csv",
                          path_data:str = "gs://starcop/Permian/data/",
                          path_data_official:str = "gs://starcop/Permian/permian_2019_official"):
    """
    I used this code to process the plumes of the Permian basin and its corresponding plume list (downloaded from
    https://zenodo.org/record/5610307#.Yp83JhxBzRZ)

    Args:
        name_file: path to plume list
        filepath_out: name of csv file out with path to files
        path_data: where AVIRIS-NG products are downloaded
        path_data_official: Where the images downloaded from zenodo are stored

    Returns:

    """
    import fsspec
    from georeader.rasterio_reader import RasterioReader

    dataframe = pd.read_csv(name_file)

    # Create name (AVIRIS-NG product name)
    dataframe["name"] = dataframe.candidate_id.apply(lambda x: x[:18])

    # Add path to folder with products
    dataframe["folder"] = dataframe.name.apply(lambda x: os.path.join(path_data, x))
    fs = fsspec.filesystem("gs")

    # gs://starcop/Permian/permian_2019_official/ang20190922t192642-2_r4578_c217_ctr.tif
    dataframe["label_path"] = dataframe.candidate_id.apply(
        lambda x: "gs://" + fs.glob(f"{path_data_official}/{x}_r*_ctr.tif")[0])
    dataframe["rgb_path"] = dataframe.label_path.apply(lambda x: x.replace("_ctr.tif", "_rgb.tif"))

    readers = {}  # cache of readers to avoid re-reading
    windows = []
    for row in tqdm(dataframe.itertuples(), total=dataframe.shape[0]):
        mag1c_product = row.folder + "mag1c.tif"
        if not fs.exists(mag1c_product):
            windows.append(None)
            continue

        label_reader = RasterioReader(row.label_path)
        if not mag1c_product in readers:
            readers[mag1c_product] = RasterioReader(mag1c_product)

        window = window_from_product(readers[mag1c_product], label_reader)
        windows.append(window)

    dataframe["window"] = windows

    # Serialize windows to save
    for attr_name in ["col_off", "row_off", "width", "height"]:
        dataframe[f"window_{attr_name}"] = dataframe["window"].apply(
            lambda x: None if x is None else getattr(x, attr_name))
        assert dataframe[f"window_{attr_name}"][~dataframe.window_col_off.isna()].apply(_is_exact_round).all()
        dataframe[f"window_{attr_name}"] = dataframe[f"window_{attr_name}"].fillna(-9999).astype(np.int64)

    dataframe[[c for c in dataframe.columns if c != "window"]].to_csv(
        filepath_out, index=False)