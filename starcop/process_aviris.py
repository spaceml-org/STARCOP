from typing import Optional, List, Dict
from starcop.utils import get_filesystem, download_product, write_json_to_gcp, read_json_from_gcp
from starcop.data import aviris
import os
import numpy as np
import rasterio
import rasterio.windows
import pandas as pd
from glob import glob
import subprocess
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from georeader.save_cog import save_cog
from tqdm import tqdm
import spectral

from starcop.models import mag1c
import torch


BANDS_S2 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', "B8A", 'B9', 'B10', 'B11', 'B12']
BANDS_WV3 = ["SWIR1","SWIR2", "SWIR3", "SWIR4","SWIR5","SWIR6","SWIR7","SWIR8"]

BANDS_SENSOR = {"S2A":BANDS_S2, "S2B": BANDS_S2, "WV3": BANDS_WV3}

def aviris_as_sensor(aviris_img_folder_or_path:str,
                     folder_dest:str,
                     sensors:List[str]=list(BANDS_SENSOR.keys()),
                     bands:Dict[str,List[str]]=BANDS_SENSOR,
                     columns_read:int=50,
                     path_tiffs_temp:str=".",
                     disable_pbar=True):
    """
    Aggregate the bands of AVIRIS according to the spectral response function of Sentinel-2. This process produce
    an image with each of the corresponding bands as observed by S2A or S2B.

    Args:
        aviris_img_folder_or_path:
        folder_dest:
        sensors:
        bands:
        columns_read:
        path_tiffs_temp:
        disable_pbar:

    Returns:

    """

    if not folder_dest.startswith("gs://"):
        os.makedirs(folder_dest, exist_ok=True)

    # Read AVIRIS image
    rdn, windows = aviris.read_aviris(aviris_img_folder_or_path, columns_read=columns_read, return_windows=True)

    fs_dest = get_filesystem(folder_dest)

    for sensor in sensors:
        for band in bands[sensor]:
            dst_file = os.path.join(folder_dest, f"{sensor}_{band}.tif")
            if fs_dest.exists(dst_file):
                continue

            aviris_sensor_band_np = np.zeros(rdn.shape[1:], dtype=np.float32)

            # Process image by windows to limit memory consumption
            for wv in tqdm(windows, desc=f"Processing band {band} for sensor {sensor} by tiles",
                           disable=disable_pbar):
                rdn_wv = rdn.read_from_window(wv)
                if sensor.startswith("S2"):
                    aviris_sensor_band_col = aviris.transform_to_sentinel_2(rdn_wv, bands_s2=[band],
                                                                            resolution_dst=None, sensor=sensor,
                                                                            verbose=False,
                                                                            fill_value_default=rdn_wv.fill_value_default)
                elif sensor == "WV3":
                    aviris_sensor_band_col = aviris.transform_to_worldview_3(rdn_wv, bands_wv3=[band],
                                                                            resolution_dst=None,
                                                                            verbose=False,
                                                                            fill_value_default=rdn_wv.fill_value_default)
                else:
                    raise NotImplementedError(f"Sensor {sensor} not known. Expected sensors [WV3, S2A, S2B]")


                aviris_sensor_band_np[wv.toslices()] = aviris_sensor_band_col.values[0]

            aviris_sensor_band = GeoTensor(aviris_sensor_band_np, transform=rdn.transform, crs=rdn.crs,
                                           fill_value_default=rdn.fill_value_default)

            save_cog(aviris_sensor_band, dst_file, descriptions=[band],
                     dir_tmpfiles=path_tiffs_temp)


def save_aviris_cog(aviris_img_folder:str, folder_dest:str, path_tiffs_temp:str= ".",
                    disable_pbar:bool=False):
    """
    Save each band of AVIRIS in a different COG object

    Args:
        aviris_img_folder:
        folder_dest:
        path_tiffs_temp:
        disable_pbar:

    Returns:

    """

    fs = get_filesystem(folder_dest)

    if aviris_img_folder.endswith("/"):
        aviris_img_folder = aviris_img_folder[:-1]

    name_img = os.path.basename(aviris_img_folder)
    rdn_path = os.path.join(aviris_img_folder, f"{name_img}_img")
    rdn = RasterioReader(rdn_path)

    rdn_file = spectral.io.envi.open(rdn_path + '.hdr')
    filename_dest_json = os.path.join(folder_dest, "metadata.json")

    if not fs.exists(filename_dest_json):
        write_json_to_gcp(filename_dest_json, {"wavelengths": rdn_file.bands.centers,
                                               "bandwidths": rdn_file.bands.bandwidths})

    desc = rdn.descriptions
    for bidx in tqdm(range(rdn.shape[0]), desc="Creating COGs in bucket", total=rdn.shape[0],
                     disable=disable_pbar):
        filename_dest_band = os.path.join(folder_dest, f"{bidx}.tif")
        if fs.exists(filename_dest_band):
            continue

        band = rdn.isel({"band": slice(bidx, bidx + 1)})
        save_cog(band, filename_dest_band,
                 descriptions=[desc[bidx]], dir_tmpfiles=path_tiffs_temp)


def run_mag1c(aviris_img_folder:str,
              mf_filename:str,
              albedo_filename:Optional[str]=None,
              glt_filename:Optional[str]=None,
              use_wavelength_range=(2122, 2488),
              disable_pbar:bool=False,
              device="cpu",
              path_tiffs_temp:str=".",
              samples_read:int=50):
    """
    Run mag1c in an AVIRIS image and save its output as a COG GeoTIFF

    Args:
        aviris_img_folder: Folder with the original AVIRIS product
        mf_filename: path to save to matching filter output (expected .tif extension)
        albedo_filename: Optional path to save the albedo (expected .tif extension)
        glt_filename: Optional path to save the glt file (expected .tif extension)
        use_wavelength_range: Defines what contiguous range of wavelengths (in nanometers) should be included in the
        filter calculation
        disable_pbar: do not show pbar
        device: string with device for torch
        path_tiffs_temp:
        samples_read:

    Returns:

    """
    device = torch.device(device)

    fs = get_filesystem(mf_filename)
    if fs.exists(mf_filename) and ((albedo_filename is None) or fs.exists(albedo_filename)) and ((glt_filename is None) or fs.exists(glt_filename)):
        return

    if aviris_img_folder.endswith("/"):
        aviris_img_folder = aviris_img_folder[:-1]

    name_img = os.path.basename(aviris_img_folder)
    rdnfromgeo = os.path.join(aviris_img_folder, f"{name_img}_glt")
    assert os.path.exists(rdnfromgeo), f"File {rdnfromgeo} does not exists"

    rdn = os.path.join(aviris_img_folder, f"{name_img}_img")
    assert os.path.exists(rdn), f"File {rdnfromgeo} does not exists"

    with rasterio.open(rdn) as src:
        transform = src.transform
        crs = src.crs

    rdn_file = spectral.io.envi.open(rdn + '.hdr')
    rdn_file_memmap = rdn_file.open_memmap(interleave='bip', writable=False)

    src_glt_file = spectral.io.envi.open(rdnfromgeo + '.hdr')
    src_glt_file_memmap = src_glt_file.open_memmap(interleave='bip', writable=False)

    wavelengths = np.array(rdn_file.bands.centers)

    # Bands to keep are those wavelengths not affected by water vapour
    band_keep = mag1c.get_mask_bad_bands(wavelengths)
    band_keep[wavelengths < use_wavelength_range[0]] = False
    band_keep[wavelengths > use_wavelength_range[1]] = False
    wave_keep = wavelengths[band_keep]

    # target spectrum
    target = mag1c.generate_template_from_bands(centers=rdn_file.bands.centers, fwhm=rdn_file.bands.bandwidths)
    target = target.astype(rdn_file_memmap.dtype)

    spec = torch.tensor(target[band_keep, 1], device=device)

    idx_keep, = np.where(band_keep)
    assert idx_keep[-1] - idx_keep[0] + 1 == idx_keep.shape[0], "Not all indexes included. Can't be a slice!"
    # Slice to read, spatial plus bands
    slice_bands = slice(idx_keep[0], idx_keep[-1] + 1)

    # Run mag1c only if needed
    if not fs.exists(mf_filename) or ((albedo_filename is not None) and not fs.exists(albedo_filename)):
        function_acfwl1mf = lambda x: mag1c.acrwl1mf(torch.as_tensor(x, device=device), spec, num_iter=30)
        samples_glt_file = np.abs(np.array(src_glt_file_memmap[..., 0]))
        valid_mask = samples_glt_file != 0

        rdn_data = rdn_file_memmap[..., slice_bands]

        print(f"Processing image {name_img} of size: {rdn_data.shape}")
        mf_out, albedo_out = mag1c.func_by_groups(function_acfwl1mf, rdn_data, samples_glt_file,
                                                  mask=valid_mask,disable_pbar=disable_pbar,
                                                  samples_read=samples_read)

        mf_out = GeoTensor(mf_out, transform=transform, crs=crs, fill_value_default=mag1c.NODATA)
        # albedo_out = GeoTensor(albedo_out, transform=transform, crs=crs, fill_value_default=NODATA)

        save_cog(mf_out, mf_filename, descriptions=["CH4 Absorption (ppm x m)"],
                 tags={"wavelengths": wave_keep,
                       "mag1c": "acfwl1mf"},
                 dir_tmpfiles=path_tiffs_temp)

        if albedo_filename is not None:
            albedo_out = GeoTensor(albedo_out, transform=transform, crs=crs, fill_value_default=mag1c.NODATA)
            save_cog(albedo_out, albedo_filename, descriptions=["Albedo"],
                     tags={"wavelengths": wave_keep,
                           "mag1c": "acfwl1mf"},
                     dir_tmpfiles=path_tiffs_temp)

    # Save glt also as COG for future use
    if glt_filename is not None:
        glt_out = GeoTensor(np.transpose(np.array(src_glt_file_memmap),(2,0,1)),
                            transform=transform, crs=crs, fill_value_default=0)
        save_cog(glt_out, glt_filename, descriptions=['GLT Sample Lookup', 'GLT Line Lookup'],
                 dir_tmpfiles=path_tiffs_temp)


def download_aviris(name:str, path_targz_base:Optional[str] = None,
                    path_untar_folder_base: Optional[str]= None,
                    display_progress_bar_download:bool=True,
                    remove_targz_file:bool=True):
    f"""
        This function downloads an AVIRIS-NG image in the directory `path_targz_base` and unzips it in the directory
        `path_untar_folder_base`. It returns the path to the downloaded zipped and unzziped files.

        Args:
            name: AVIRIS-NG image in format: angYYYYmmddtHHMMSS (e.g. ang20150419t194538)
            path_targz_base: dir to save the downloaded file (.tar.gz)
            path_untar_folder_base: dir to unzip the product
            remove_targz_file: whether or not to remove the intermediate files
            display_progress_bar_download: 

        Returns:
            filename_down, path_untar_folder
            path to the downloaded zipped and unzziped folder.
        """


    data = pd.read_csv('gs://starcop/AVIRIS-NG-Flight-Lines.csv')
    data = data.set_index("Name")

    # Download tar.gz
    if name not in data.index:
        print(f"\t ERROR: {name} not found in AVIRIS NG data index (file: gs://starcop/AVIRIS-NG-Flight-Lines.csv)")
        return False

    link_download = data.loc[name, "link_ftp"]
    if not isinstance(link_download, str):
        print(f"\t ERROR: Link not recognized {link_download} for file {name}")
        return False

    if path_targz_base is None:
        path_targz_base = "."
    if path_untar_folder_base is None:
        path_untar_folder_base = "."

    filename_down = os.path.join(path_targz_base, os.path.basename(link_download))
    names_img_in_untar_folder = glob(os.path.join(path_untar_folder_base, f"{name}_rdn_*", f"{name}_rdn_*_img"))
    if len(names_img_in_untar_folder) > 0:
        print(f"AVIRIS untar files exists. It wont download them again")
        return filename_down, os.path.dirname(names_img_in_untar_folder[0])

    if not os.path.exists(filename_down):
        print(f"\t Downloading product {name}")
        download_product(link_download, filename=filename_down,
                         display_progress_bar=display_progress_bar_download)

    if not os.path.exists(filename_down):
        print(f"\t ERROR product {name} does not have been downloaded in {filename_down}")
        return False

    # Untar tar.gz
    names_img_in_untar_folder = glob(os.path.join(path_untar_folder_base, f"{name}_rdn_*", f"{name}_rdn_*_img"))

    if len(names_img_in_untar_folder) == 0:
        print(f"\t Untar product {name}")
        subprocess.run(["tar", "-xvzf", filename_down, "-C", path_untar_folder_base])

    names_untar_folder = glob(os.path.join(path_untar_folder_base, f"{name}_rdn_*"))
    if len(names_untar_folder) != 1:
        print(f"ERROR untar folders not correctly resolved {names_untar_folder}")

    if remove_targz_file and os.path.exists(filename_down):
        os.remove(filename_down)

    return filename_down, names_untar_folder[0]
