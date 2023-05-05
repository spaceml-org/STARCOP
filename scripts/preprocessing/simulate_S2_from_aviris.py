import os, glob
import sys
sys.path.append("..")

from starcop.data import aviris
import pandas as pd
import os
import numpy as np
import fsspec
from georeader import rasterio_reader, read
import geopandas as gpd

from georeader.rasterio_reader import RasterioReader
from georeader import read
from georeader import save_cog

source_folder = "../../datasets/PB_dataset_min200_padmin20"
name_appendix = "_allbands.tif" # to id the AVIRIS tiles
output_folder = "../../datasets/PB_SimulatedS2_min200_padmin20"

os.makedirs(output_folder, exist_ok=True)

# Conversion parameters:
sensor = "S2A" # < we could generate both and consider them as an data aug
BANDS_S2 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',"B8A", 'B9', 'B10', 'B11', 'B12']
bands = BANDS_S2
# resolution_dst=10
resolution_dst=None

convert_appendix = "_simulated_"+sensor+".tif"

files_to_process = glob.glob(source_folder+"/*"+name_appendix)
files_to_process.sort()

for idx, file in enumerate(files_to_process):
    # if idx < 149:
    #     # skip processed files ...
    #     # print("skipping", idx)
    #     continue
        
    file_name = file.split("/")[-1]
    output_file_name = file_name.replace(name_appendix,convert_appendix)
    print("Processing", idx, "/", len(files_to_process),":", file_name, "will generate", output_file_name)

    reader_aviris = RasterioReader(file)
    bands_nanometers_aviris_over = reader_aviris.descriptions
    loaded_aviris_full_tile = reader_aviris.load()
    loaded_aviris_full_tile.descriptions = bands_nanometers_aviris_over
    
    print(loaded_aviris_full_tile)
    
    try:
    
        aviris_as_s2 = aviris.transform_to_sentinel_2(loaded_aviris_full_tile, bands_s2=bands,
                                                           resolution_dst=resolution_dst,sensor=sensor,
                                                           verbose=True)

        print(aviris_as_s2)

        save_path = output_folder+"/"+output_file_name
        print("Pre-save:", save_path)
        save_cog.save_cog(aviris_as_s2, save_path, descriptions=bands)
        print("Saved into:", save_path)

    except Exception as e: 
        print("FAILED WITH:", save_path)
        print(e)

        
