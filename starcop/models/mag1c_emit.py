# This code is an adapted version from Rutzika et al 2023:
# https://colab.research.google.com/drive/1hVcLoY3R0QryLVSLG6C4jTSG-4Jgg_4c?usp=sharing
# The original mag1c code is here: https://github.com/markusfoote/mag1c/blob/master/mag1c/mag1c.py#LL579C71-L579C83

from georeader.readers.emit import EMITImage
from georeader.geotensor import GeoTensor
from . import mag1c
import torch
from typing import Optional, Tuple, Union
import numpy as np
from tqdm import tqdm


DEFAULT_WAVELENGTH_RANGE = (2122, 2488)

def mag1c_emit(ei:EMITImage, device:Optional[torch.device]=torch.device('cpu'), 
               use_wavelength_range = DEFAULT_WAVELENGTH_RANGE,
               num_iter:int=30, covariance_lerp_alpha:float=1e-4,
               column_step:Optional[int]=None,
               georreferenced:bool=True,display_pbar:bool=True) -> Tuple[Union[GeoTensor,np.array], Union[GeoTensor,np.array]]:
    """
    Run mag1c filter on an EMITImage object.

    Args:
        ei (EMITImage): EMITImage object
        device (Optional[torch.device], optional): device to use for the computation. Defaults to torch.device('cpu').
        use_wavelength_range (tuple, optional): wavelength range to use. Defaults to (2000, 2485).
        num_iter (int, optional): Number of iterations. Defaults to 30.
        covariance_lerp_alpha (float, optional): parameter to control the covariance matrix. Defaults to 1e-4.
        column_step (Optional[int], optional): column step to use. Defaults to None. Column step 1 runs the
            mag1c filter by column. None runs the mag1c filter in all the image.
        georreferenced (bool, optional): If True, the output is a GeoTensor. Defaults to True.
        display_pbar (bool, optional): If True, display a progress bar. Defaults to True.

    Returns:
        Tuple[Union[GeoTensor,np.array], Union[GeoTensor,np.array]]: Tuple with the mag1c filter and the albedo.

    """

    band_selection = (ei.wavelengths >= use_wavelength_range[0]) & (ei.wavelengths <= use_wavelength_range[1])
    assert band_selection.any(), "There are no bands in the selected wavelength range"

    ei = ei.read_from_bands(band_selection)

    target = mag1c.generate_template_from_bands(centers=ei.wavelengths, fwhm=ei.fwhm)
    target = target.astype(np.float64)

    spec = torch.tensor(target[:, 1], device=device)

    raw_data = ei.load_raw(transpose=False) # (rows, cols, bands)
    invalid = np.any(raw_data == ei.fill_value_default, axis=-1) # (rows, cols)

    mag1c_output = np.full(invalid.shape, dtype=np.float64, fill_value=ei.fill_value_default) # (rows, cols)
    albedo_out = np.full(invalid.shape, dtype=np.float64, fill_value=ei.fill_value_default) # (rows, cols)
    
    column_step = column_step or raw_data.shape[1]

    listrange = range(0, raw_data.shape[1], column_step)
    for columnstart in tqdm(listrange,
                            total=len(listrange),
                            desc="\tRunning mag1c filter by columns",
                            disable=not display_pbar or (column_step==raw_data.shape[1])):
        columnend = min(columnstart + column_step, raw_data.shape[1])
        invalid_slice = invalid[:, columnstart:columnend] # (rows, column_step)
        raw_data_slice = raw_data[:, columnstart:columnend, :] # (rows, column_step, bands)

        valid_slice = ~invalid_slice
        if not valid_slice.any():
            continue
    
        raw_data_slice = raw_data_slice[valid_slice, :][np.newaxis] # (batch, pixels, bands)
        # pixels is the number of valid pixels (np.sum(valid_slice))

        # Convert to float64 to avoid rounding errors
        raw_data_slice = raw_data_slice.astype(np.float64)
        raw_data_slice = torch.tensor(raw_data_slice, device=device)

        mf_out, alb_out = mag1c.acrwl1mf(raw_data_slice, spec, num_iter=num_iter, alpha=covariance_lerp_alpha)

        mf_out = np.array(mf_out.cpu())[0,:,0] # (pixels,)
        alb_out = np.array(alb_out.cpu())[0,:,0] # (pixels,)

        mag1c_output[:, columnstart:columnend][valid_slice] = mf_out
        albedo_out[:, columnstart:columnend][valid_slice] = alb_out

    if georreferenced:
        mag1c_output = ei.georreference(mag1c_output, fill_value_default=ei.fill_value_default)
        albedo_out = ei.georreference(albedo_out, fill_value_default=ei.fill_value_default)

    return mag1c_output.astype(np.float32), albedo_out.astype(np.float32)

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import netCDF4
# from spectral.io import envi
# import os


# def run_magic_on_data(data, slice_for_data, target_spec, nodata = -9999.0, show=True, save=False, figsize=20,
#                       num_iter = 30, covariance_lerp_alpha = 1e-4):
#     # Using code snippets from https://github.com/spaceml-org/STARCOP
#     # specifically this notebook: https://github.com/spaceml-org/STARCOP/blob/training-lightning/notebooks/exploration_aviris_mag1c.ipynb
#     print("Reading data", data.shape," with slice: ", slice_for_data)
#     print("Target:", target_spec.shape, target_spec.dtype)
#     rdn_data = np.array(data[slice_for_data]).astype(np.float64)
#     print("Sliced into:", rdn_data.shape, rdn_data.dtype)

#     valid_mask = np.where(np.array(data[:, :, 0]) == nodata, False, True)
#     valid_mask_slice = valid_mask[slice_for_data[0:2]]
#     rdn_data_valid = rdn_data[valid_mask_slice,:]
#     rdn_data_valid_torch = torch.Tensor(rdn_data_valid).unsqueeze(0) # [b, p, s]

#     mf_out, albedo_out = mag1c.acrwl1mf(rdn_data_valid_torch, target_spec, num_iter=num_iter, alpha=covariance_lerp_alpha)

#     magic_output = np.zeros(rdn_data.shape[:2],dtype=rdn_data.dtype) + nodata
#     magic_output[valid_mask_slice] = np.array(mf_out)[0,:,0]

#     plt.figure(figsize=(figsize,figsize),tight_layout=True)
#     plt.imshow(magic_output, vmin=0, vmax=5000)
#     plt.title("mag1c on a full tile modified")

#     if save is not False:
#         plt.savefig(save)

#     if show:
#         plt.show()

#     return magic_output

# def prep_emit_data(data, wavelengths, bandwidths):
#     wavelengths = np.array(wavelengths)
#     bandwidths = np.array(bandwidths)
    

#     # for tests we can also use just 5 instead of the around 8 that EMIT has with:
#     # faked_bandwidths = 5.0 * np.ones_like(bandwidths)
#     # bandwidths = faked_bandwidths

#     band_keep = mag1c.get_mask_bad_bands(wavelengths)
#     use_wavelength_range = (2122, 2488)  # defaults in mag1c
#     use_wavelength_range = (2000, 2488)  # increase range
#     # Remove bands out of the absortion range of methane
#     band_keep[wavelengths < use_wavelength_range[0]] = False
#     band_keep[wavelengths > use_wavelength_range[1]] = False
#     idx_keep, = np.where(band_keep)  # Which bands to keep -> 72 bands

#     target = mag1c.generate_template_from_bands(centers=wavelengths, fwhm=bandwidths)
#     # plt.figure(figsize=(10, 5))
#     # plt.plot(target[:, 0], target[:, 1])
#     # plt.vlines(use_wavelength_range, ymin=-1.5, ymax=0, color="C3")
#     # plt.show()

#     target_spec = torch.from_numpy(target[band_keep, 1]).to(device=torch.device("cpu"), dtype=torch.float32)

#     ### Slice data:

#     slice_ = (slice(None),slice(None)) # all, but slow!
#     slice_bands = slice(idx_keep[0], idx_keep[-1] + 1)
#     slice_with_bands = slice_ + (slice_bands,)

#     return data, slice_with_bands, target_spec, wavelengths


# # Our own code, which was edited
# # from a emit data processing source at: https://github.com/emit-sds/emit-utils/blob/develop/emit_utils/reformat.py

# def envi_header(inputpath):
#     """
#     Convert a envi binary/header path to a header, handling extensions
#     Args:
#         inputpath: path to envi binary file
#     Returns:
#         str: the header file associated with the input reference.
#     """
#     if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
#         # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
#         # does, if not return the latter (new file creation presumed).
#         hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
#         if os.path.isfile(hdrfile):
#             return hdrfile
#         elif os.path.isfile(inputpath + '.hdr'):
#             return inputpath + '.hdr'
#         return hdrfile
#     elif os.path.splitext(inputpath)[-1] == '.hdr':
#         return inputpath
#     else:
#         return inputpath + '.hdr'


# envi_typemap = {
#     'uint8': 1,
#     'int16': 2,
#     'int32': 3,
#     'float32': 4,
#     'float64': 5,
#     'complex64': 6,
#     'complex128': 9,
#     'uint16': 12,
#     'uint32': 13,
#     'int64': 14,
#     'uint64': 15
# }


# def single_image_ortho(img_dat, glt, glt_nodata_value=0, set_nodata=np.nan):
#     """Orthorectify a single image

#     Args:
#         img_dat (array like): raw input image
#         glt (array like): glt - 2 band 1-based indexing for output file(x, y)
#         glt_nodata_value (int, optional): Value from glt to ignore. Defaults to 0.

#     Returns:
#         array like: orthorectified version of img_dat
#     """
#     outdat = np.ones((glt.shape[0], glt.shape[1], img_dat.shape[-1])) * set_nodata
#     valid_glt = np.all(glt != glt_nodata_value, axis=-1)
#     glt[valid_glt] -= 1  # account for 1-based indexing
#     outdat[valid_glt, :] = img_dat[glt[valid_glt, 1], glt[valid_glt, 0], :]
#     return outdat

# def load_nc_file(input_netcdf = "what.nc", output_dir="", output_type='ENVI', interleave='BIL', overwrite=False):
#     nc_ds = netCDF4.Dataset(input_netcdf, 'r', format='NETCDF4')
#     orthorectify_params = {}

#     glt = np.zeros(list(nc_ds.groups['location']['glt_x'].shape) + [2], dtype=np.int32)
#     glt[..., 0] = np.array(nc_ds.groups['location']['glt_x'])
#     glt[..., 1] = np.array(nc_ds.groups['location']['glt_y'])

#     if output_type == 'ENVI':
#         dataset_names = list(nc_ds.variables.keys())
#         for ds in dataset_names:
#             output_name = os.path.join(output_dir, os.path.splitext(os.path.basename(input_netcdf))[0] + '_' + ds)
#             if os.path.isfile(output_name+"_RGB") and os.path.isfile(output_name+"_magic") and overwrite is False:
#                 print(f'File {output_name} already exists. Skipping!')
#                 return None, None, None, None

#             metadata = {
#                 'lines': nc_ds[ds].shape[0],
#                 'samples': nc_ds[ds].shape[1],
#                 'bands': nc_ds[ds].shape[2],
#                 'interleave': interleave,
#                 'header offset': 0,
#                 'file type': 'ENVI Standard',
#                 'data type': envi_typemap[str(nc_ds[ds].dtype)],
#                 'byte order': 0
#             }

#             for key in list(nc_ds.__dict__.keys()):
#                 if key == 'summary':
#                     metadata['description'] = nc_ds.__dict__[key]
#                 elif key not in ['geotransform', 'spatial_ref']:
#                     metadata[key] = f'{{ {nc_ds.__dict__[key]} }}'

#             orthorectify_params["glt"] = glt
#             gt = np.array(nc_ds.__dict__["geotransform"])
#             orthorectify_params["map info"] = f'{{Geographic Lat/Lon, 1, 1, {gt[0]}, {gt[3]}, {gt[1]}, {gt[5] * -1},WGS-84}}'
#             orthorectify_params["coordinate system string"] = f'{{ {nc_ds.__dict__["spatial_ref"]} }}'

#             band_parameters = nc_ds['sensor_band_parameters'].variables.keys()
#             for bp in band_parameters:
#                 if bp == 'wavelengths' or bp == 'radiance_wl':
#                     metadata['wavelength'] = np.array(nc_ds['sensor_band_parameters'].variables[bp]).astype(
#                         str).tolist()
#                 elif bp == 'radiance_fwhm':
#                     metadata['fwhm'] = np.array(nc_ds['sensor_band_parameters'].variables[bp]).astype(str).tolist()
#                 elif bp == 'observation_bands':
#                     metadata['band names'] = np.array(nc_ds['sensor_band_parameters'].variables[bp]).astype(
#                         str).tolist()
#                 else:
#                     metadata[bp] = np.array(nc_ds['sensor_band_parameters'].variables[bp]).astype(str).tolist()

#             if 'wavelength' in list(metadata.keys()) and 'band names' not in list(metadata.keys()):
#                 metadata['band names'] = metadata['wavelength']

#             data = np.array(nc_ds[ds])
#     return data, metadata, output_name, orthorectify_params

# def orthorectify_data(data, metadata, orthorectify_params):
#     metadata['lines'] = orthorectify_params["glt"].shape[0]
#     metadata['samples'] = orthorectify_params["glt"].shape[1]
#     metadata['map info'] = orthorectify_params["map info"]
#     metadata['coordinate system string'] = orthorectify_params["coordinate system string"]
#     data = single_image_ortho(data, orthorectify_params["glt"])
#     return data, metadata


# def reformat_nc_file_into_RGB(input_netcdf = "what.nc", output_dir="", output_type='ENVI', interleave='BIL',
#                               orthorectify=True, overwrite=True, no_data_to_nans = True):
#     # if overwrite is set to False, we will skip already existing files...
#     print("File:", input_netcdf.split("/")[-1])
#     if os.path.isdir(output_dir) is False:
#         err_str = f'Output directory {output_dir} does not exist - please create or try again'
#         raise AttributeError(err_str)

#     # 01 LOAD RAW DATA
#     data, metadata, output_name, orthorectify_params = load_nc_file(input_netcdf, output_dir, output_type, interleave, overwrite)
#     if data is None:
#         return None # skipped

#     wavelengths = np.array(metadata['wavelength']).astype(float).tolist()
#     bandwidths = np.array(metadata['fwhm']).astype(float).tolist()
#     print("We have ", len(wavelengths), "wavelengths, from", wavelengths[0], " nm to", wavelengths[-1], " nm...")
#     print("We have ", len(bandwidths), "bandwidths, from", bandwidths[0], " nm to", bandwidths[-1], " nm...")


#     # 02 SO SOMETHING WITH IT ...
#     # ... to do ... mag1c ?
#     data, slice_for_data, target_spec, _ = prep_emit_data(data, wavelengths, bandwidths)
#     magic_output = run_magic_on_data(data=data, slice_for_data=slice_for_data, target_spec=target_spec,
#                       show=False, nodata=-9999.0)
#                       # save=p + "_magic_output_v0_withalphacov0.jpg", show=False)
#     # set the mag1c no data to np.nan (for better visualisation...)
#     if no_data_to_nans:
#         magic_output = np.where(magic_output == -9999.0, np.nan, magic_output)

#     # 03 ORTHORECTIFY AND GEOTAG THE RESULT

#     rgb_bands = [36, 22, 10, 0] # note the 4th band will be rewritten ...
#     data = data[:, :, rgb_bands]
#     data[:, :, -1] = magic_output # < we overwrite the 0th band

#     if orthorectify:
#         data, metadata = orthorectify_data(data, metadata, orthorectify_params)

#     # 04 SAVE
#     # RGB
#     rgb = data[:,:,[0,1,2]]
#     if no_data_to_nans:
#         rgb = np.where(rgb == -9999.0, np.nan, rgb)

#     magic = data[:,:,[3]]

#     metadata['bands'] = 3  # RGB
#     file_name = output_name + "_RGB"
#     envi_ds = envi.create_image(envi_header(file_name), metadata, ext='', force=True)
#     mm = envi_ds.open_memmap(interleave='bip', writable=True)
#     mm[...] = rgb
#     del mm, envi_ds # flush?

#     metadata['bands'] = 1  # mag1c
#     file_name = output_name + "_magic"
#     envi_ds = envi.create_image(envi_header(file_name), metadata, ext='', force=True)
#     mm = envi_ds.open_memmap(interleave='bip', writable=True)
#     mm[...] = magic
#     del mm, envi_ds
