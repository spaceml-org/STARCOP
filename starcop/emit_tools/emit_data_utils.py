import os
from glob import glob
import numpy as np
import rasterio

def load_emit_dataset(emit_dataset_folder = '../../EMIT_DATASET', labels_name = "label.tif", verbose = False):
    # note: labels_name influences which labels we load
    #  in our data "label.tif" marks our manual, and "label_released.tif" marks the official, but coarser ones
    positive_files = sorted(glob(os.path.join(emit_dataset_folder+"/plume_events","*")))
    negative_files = sorted(glob(os.path.join(emit_dataset_folder+"/confounders","*")))
    if verbose: print(len(positive_files), positive_files)
    if verbose: print(len(negative_files), negative_files)

    all_files = positive_files + negative_files
    all_files = [a for a in all_files if os.path.isdir(a)]
    if verbose: print("All:",len(all_files), all_files)

    dataset_paths = []

    for one_location in all_files:
        subfiles = glob(one_location+"/*")

        rgbs = [f for f in subfiles if ("RGB" in f and ".hdr" not in f)]
        magics = [f.replace("_RGB", "_magic") for f in rgbs]
        if verbose: print("we have ", len(rgbs), "rgb files and accompanying", len(magics), "magic files")

        # do we have a label?
        we_have_label = os.path.isfile(one_location+"/"+labels_name) 

        rgb_p = rgbs[0]
        magic_p = magics[0]
        label_p = None
        if we_have_label:
            label_p = one_location+"/"+labels_name

        dataset_paths.append([rgb_p,magic_p,label_p])

    if verbose: print("The dataset contains", len(dataset_paths))
    
    return dataset_paths


def load_data(dataset_paths, load_products="all"):

    data = []
    for i in range(len(dataset_paths)):
        rgb_p, magic_p, label_p = dataset_paths[i]
        name = rgb_p.split("/")[-1].replace("_radiance_RGB", "")

        if load_products != "mag1c_only":
            with rasterio.open(rgb_p) as src:
                src_rgb = src.read()

        with rasterio.open(magic_p) as src:
            src_magic = src.read()
            magic_data = src_magic[0]

        #if nan_to_num:
        #    src_rgb = np.nan_to_num(src_rgb, nan=1.0)

        if label_p is None:
            label_data = np.zeros_like(magic_data)
        else:
            with rasterio.open(label_p) as src:
                label_data = src.read()
            label_data = label_data[0]

        if load_products != "mag1c_only":
            data.append([src_rgb,magic_data,label_data,rgb_p])
        else:
            data.append([magic_data,label_data,rgb_p])
                    
    #data = np.asarray(data)
    if load_products != "mag1c_only":
        print("Loaded", len(data), "samples. First one (rgb, magic, label):", data[0][0].shape, data[0][1].shape, data[0][2].shape, "paths like", data[0][3])
    else:
        print("Loaded", len(data), "samples. First one (magic, label):", data[0][0].shape, data[0][1].shape, "paths like", data[0][2])
    return data
