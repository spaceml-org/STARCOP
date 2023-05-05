import sys

sys.path.append("..")

from starcop.data import sampling_dataset
from tqdm import tqdm
from georeader import save_cog
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
from starcop.utils import get_filesystem
import os
import numpy as np

dataframe = sampling_dataset.permian_plumes_dataframe()

aviris_products = dataframe["name"].unique()
aviris_products.sort()

for name in tqdm(aviris_products):
    plumes_name = dataframe[dataframe["name"] == name]
    folder_name = plumes_name.iloc[0].folder
    output_labels_path = os.path.join(folder_name, "label_rgba.tif")
    fs = get_filesystem(output_labels_path)
    if fs.exists(output_labels_path):
        continue
    first_band_path = os.path.join(folder_name, "1.tif")
    first_band = RasterioReader(first_band_path)

    output_product = GeoTensor(np.zeros((4,) + first_band.shape[-2:], dtype=np.uint8),
                               transform=first_band.transform,
                               crs=first_band.crs, fill_value_default=0)

    for plume_iter in plumes_name.itertuples():
        reader = RasterioReader(plume_iter.label_path, fill_value_default=0)
        output_product.write_from_window(reader.values, plume_iter.window)

    save_cog.save_cog(output_product, output_labels_path, descriptions=["R", "G", "B", "A"])

