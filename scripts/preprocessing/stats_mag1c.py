import os
from starcop.utils import get_filesystem
from georeader.rasterio_reader import RasterioReader
import pandas as pd
import numpy as np
from tqdm import tqdm
from georeader import slices

def percentile05(x):
    return np.percentile(x, 5)

def percentile95(x):
    return np.percentile(x, 95)

def percentile01(x):
    return np.percentile(x, 1)

def percentile99(x):
    return np.percentile(x, 99)

def count(x):
    return x.shape[0]

if __name__ == "__main__":
    folder_save = "gs://starcop/Permian/data"
    fs = get_filesystem(folder_save)
    overwrite = True
    filename_full_out = "gs://starcop/Permian/stats_mag1c.csv"
    files = fs.glob(os.path.join(folder_save, "*", "mag1c.tif"))
    data_out = []
    for _i, f in enumerate(files):
        reader = RasterioReader(f"gs://{f}")
        folder = os.path.dirname(f"gs://{f}")
        stats_mag1c = []
        file_out = os.path.join(folder, "stats_mag1c.csv")
        if not overwrite and fs.exists(file_out):
            stats_mag1c = pd.read_csv(file_out)
            data_out.append(stats_mag1c)
            continue
        
        windows = slices.create_windows(reader, window_size=(512,512),overlap=(256, 256))
        for w in tqdm(windows,desc=f"{_i+1}/{len(files)} Processing {reader.paths[0]}"):
            data_window = reader.read_from_window(window=w).load()

            values_stats = data_window.values[data_window.values != data_window.fill_value_default]
            values_stats = values_stats[values_stats>=0]
            
            values_stats[values_stats >= 10_000] = 10_000
            
            if values_stats.shape[0] == 0:
                continue

            attributes = {}
            for attr_name in ["col_off", "row_off", "width", "height"]:
                attributes[f"window_{attr_name}"] = getattr(w, attr_name)

            attributes["folder"] = folder

            # Compute stats
            for fun in [np.max, np.min, np.mean, percentile01, percentile05, np.median, percentile95, percentile99, np.sum, count]:
                attributes[fun.__name__] = fun(values_stats)

            stats_mag1c.append(attributes)
        stats_mag1c = pd.DataFrame(stats_mag1c)
        stats_mag1c.to_csv(file_out, index=False)

        data_out.append(stats_mag1c)

    data_out = pd.concat(data_out, ignore_index=True)
    data_out.to_csv(filename_full_out, index=False)

