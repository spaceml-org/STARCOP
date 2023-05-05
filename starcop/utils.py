from tqdm import tqdm
import os
import requests
import fsspec
import subprocess
from typing import Optional, Dict
import shutil
import pandas as pd
from glob import glob
import numpy as np
import rasterio
import json


def remove_folder(name_folder):
    # Used for deleting a folder with temporary files
    shutil.rmtree(name_folder, ignore_errors=True)


def get_filesystem(path: str):
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0], requester_pays=True)
    else:
        # use local filesystem
        return fsspec.filesystem("file")


def download_product(link_down:str, filename:Optional[str]=None, display_progress_bar:bool=True) -> str:
    if filename is None:
        filename = os.path.basename(link_down)

    if os.path.exists(filename):
        print(f"File {filename} exists. It won't be downloaded again")
        return filename

    filename_tmp = filename+".tmp"

    with requests.get(link_down, stream=True) as r_link:
        total_size_in_bytes = int(r_link.headers.get('content-length', 0))
        r_link.raise_for_status()
        block_size = 8192  # 1 Kibibyte
        if display_progress_bar:
            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,position=0,leave=True) as progress_bar:
                with open(filename_tmp, 'wb') as f:
                    for chunk in r_link.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        else:
            with open(filename_tmp, 'wb') as f:
                for chunk in r_link.iter_content(chunk_size=block_size):
                    f.write(chunk)

    shutil.move(filename_tmp, filename)

    return filename

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj_to_encode):
        from shapely.geometry import Polygon, mapping
        from datetime import datetime
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, "to_json"):
            return obj_to_encode.to_json()
        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()
        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        if isinstance(obj_to_encode, Polygon):
            return mapping(obj_to_encode)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat()
        if isinstance(obj_to_encode, datetime):
            return obj_to_encode.isoformat()
        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)


def write_json_to_gcp(gs_path: str, dict_val: dict) -> None:
    fs = get_filesystem(gs_path)

    with fs.open(gs_path, "w") as fh:
        json.dump(dict_val, fh, cls=CustomJSONEncoder)


def read_json_from_gcp(gs_path: str) ->Dict:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "r") as fh:
        my_dictionary = json.load(fh)

    return my_dictionary