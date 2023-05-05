import requests
from lxml import html
import os
import subprocess
from PIL import Image
import numpy as np
import rasterio
from georeader.vectorize import get_polygons
import geopandas as gpd
from glob import glob
from datetime import datetime
from georeader import save_cog
import pandas as pd
from starcop import utils

# gdal_translate ang20150419t155032_cmf_v1f_img ang20150419t155032_cmf_v1f_img.tif -of COG -co BLOCKSIZE=512 -co RESAMPLING=BILINEAR -co COMPRESS=DEFLATE -co NUM-THREADS=6 -co BIGTIFF=IF_SAFER


def get_links():
    r = requests.get("https://avirisng.jpl.nasa.gov/AVNG_Benchmark_Methane_Data.php")
    content_html = html.fromstring(r.content)
    links_to_download = []

    for cosa in content_html.xpath("//a"):
        if "href" not in cosa.attrib:
            continue
        href = cosa.attrib["href"]
        if href.endswith(".tar.gz"):
            links_to_download.append(href)
    return links_to_download

path_save = "/media/disk/databases/JPL-CG4-detection-2017"
links_to_download = get_links()

for _i,l in enumerate(sorted(links_to_download)):
    name_file = os.path.basename(l)
    print(f"{_i+1}/{len(links_to_download)} Downloading {name_file}")
    path_dest = os.path.join(path_save,"tar",name_file)
    utils.download_product(l, path_dest)
    
    noextname = name_file.replace(".tar.gz","")
    
    # untar
    name_untar_folder_base = os.path.join(path_save, "untar")
    name_untar_folder = os.path.join(name_untar_folder_base, noextname)
    name_file_untar = os.path.join(name_untar_folder, noextname+"_img")
    if os.path.exists(name_file_untar):
        print(f"\t {name_file_untar} exists skiping untar")
    else:
        print(f"\t untar {name_untar_folder}")
        subprocess.run(["tar","-xvzf", path_dest, "-C", name_untar_folder_base])
    
    # convert to ENVI file to COG GeoTIFF
    dest_path_tiff = os.path.join(path_save,"geotiff")
    path_tiff_file = os.path.join(dest_path_tiff, "images", noextname+"_img.tif")
    if os.path.exists(path_tiff_file):
        print(f"\t tiff {path_tiff_file} will not generate again")
    else:
        print("\t converting to COG GeoTIFF")
        subprocess.run(["gdal_translate",name_file_untar, path_tiff_file, "-of", "COG", "-co", "BLOCKSIZE=512",
                    "-co", "RESAMPLING=BILINEAR", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER"])
       
    # convert mask (png) to tiff
    mask_path = os.path.join(dest_path_tiff,"masks", noextname+"_img_mask.tif")
    
    if os.path.exists(mask_path):
        print(f"\t Mask tiff exists {mask_path} will not generate again")
    else:        
        png_path = f"{name_file_untar}_mask.png"
        if not os.path.exists(png_path):
            print(f"\t Mask for file {png_path} not exist!!!!!")
            continue
        
        pilimg = Image.open(png_path)
        mask = np.array(pilimg) 
        assert mask.shape[2] == 4, f"Unexpected shape {mask.shape}"
        
        valid_mask = np.any(mask != np.array([  0,   0,   0,  255], dtype=np.uint8), 
                            axis=-1).astype(np.uint8)
        
        mask[...,3] = valid_mask * 255
        mask = np.transpose(mask, (2,0,1))
        with rasterio.open(path_tiff_file) as rst:
            crs = rst.crs
            transform = rst.transform
            shape = rst.shape
        
        assert shape == mask.shape[1:], f"Different shapes {shape} {mask.shape[1:]}"
        print("\t Saving mask as COG GeoTIFF")
        save_cog._save_cog(mask, mask_path,
                          {"crs": crs, "transform":transform,  
                           "compress": "DEFLATE",
                           "RESAMPLING": "NEAREST",
                           "nodata": 0},
                          descriptions=["R", "G", "B", "valid"])
    
    # vectorize mask
    vector_mask_path = os.path.join(dest_path_tiff, "masks_vector", noextname+"_img_mask.gpkg")
    if os.path.exists(vector_mask_path):
        print(f"\t Vector Mask exists {vector_mask_path} will not generate again")
    else:
        print("\t Vectorize raster mask")
        with rasterio.open(mask_path) as rst:
            valid_mask = rst.read(4)
            transform = rst.transform
            crs = rst.crs
        
        valid_mask = valid_mask != 0
        if not np.any(valid_mask):
            print(f"\t File does not have valid pixels. We will not vectorize")
            continue
        
        polygons = get_polygons(valid_mask, transform=transform, min_area=1)
            
        data = gpd.GeoDataFrame({"geometry": polygons, "id": np.arange(len(polygons))}, crs=crs)
        data.to_file(vector_mask_path, driver="GPKG")

        
## Join all plumes in a single file
all_plumes_file = os.path.join(path_save, "all_plumes.gpkg")

if not os.path.exists(all_plumes_file):
    data_all = []
    for f in glob(os.path.join(path_save,"geotiff/masks_vector/*.gpkg")):
        data = gpd.read_file(f)
        if data.shape[0] > 1_000:
            print(f"Skipping {f} too many plumes seems an error!")
            continue
        
        data["file"] = os.path.basename(f)
        data["sensing_time"] = datetime.strptime(os.path.basename(f)[3:18], "%Y%m%dt%H%M%S")
        data_all.append(data)

    data_all = pd.concat(data_all, ignore_index=True)

    data_all[["geometry","file","sensing_time"]].to_file(all_plumes_file, driver="GPKG")
