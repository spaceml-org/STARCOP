import pandas as pd
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from georeader import read, window_utils
from starcop.utils import get_filesystem, read_json_from_gcp
import rasterio
import rasterio.windows
import os
from typing import List, Optional, Union, Tuple
import numpy as np
from scipy import interpolate
import numbers
from datetime import datetime
from georeader.rasterio_reader import RasterioReader

SRF_S2 = None
SRF_WV3 = None

BANDS_S2_RESOLUTION = {"B1": 60, "B2": 10,
                       "B3": 10, "B4": 10,
                       "B5": 20, "B6": 20,
                       "B7": 20, "B8": 10,
                       "B8A": 20, "B9": 60,
                       "B10": 60, "B11": 20,
                       "B12": 20}

THUILLER_SOLAR_IRRADIANCE = "gs://starcop/S2/Thuillier.csv"
SRF_WV3_FILE = "gs://starcop/WV3/WV3-SRF.csv"
SRF_S2_FILE ="gs://starcop/S2/S2-SRF_joint.csv"

SOLAR_IRRADIANCE_S2B = {'B01': 1.8742999999999999, 'B02': 1.95977, 'B03': 1.8249300000000002, 'B04': 1.5127899999999999, 'B05': 1.42578,
                        'B06': 1.29113, 'B07': 1.17557, 'B08': 1.04128, 'B8A': 0.95393, 'B09': 0.8175800000000001, 'B10': 0.36541,
                        'B11': 0.24708000000000002, 'B12': 0.08775}

SOLAR_IRRADIANCE_S2A = {'B01': 1.88469, 'B02': 1.9597200000000001, 'B03': 1.82324, 'B04': 1.51206, 'B05': 1.4246400000000001,
                        'B06': 1.28761, 'B07': 1.16208, 'B08': 1.04163, 'B8A': 0.9553200000000001, 'B09': 0.81292, 'B10': 0.36715,
                        'B11': 0.24559, 'B12': 0.08525}

SOLAR_IRRADIANCE_WV3 = {"SWIR1": 477.8728/1000, "SWIR2":263.2926/1000,
                        "SWIR3":  224.9720/1000, "SWIR4": 197.3366/1000,
                        "SWIR5": 90.3976/1000, "SWIR6":  85.0757/1000,
                        "SWIR7":76.9260/1000,  "SWIR8":   68.0897/1000}


SOLAR_IRRADIANCE = {
    "S2A": SOLAR_IRRADIANCE_S2A,
    "S2B": SOLAR_IRRADIANCE_S2B,
    "WV3": SOLAR_IRRADIANCE_WV3
}

# WV3_wavelength = [1210, 1570, 1660, 1730, 2165, 2205, 2260, 2330]

def earth_sun_distance_correction_factor(date_of_acquisition:datetime) -> float:
    """
    returns: (1-0.01673*cos(0.0172*(t-4)))

     0.0172 = 360/365.256363 * np.pi/180.
     0.01673 is the Earth eccentricity

     t = datenum(Y,M,D) - datenum(Y,1,1) + 1;

     tm_yday starts in 1
     > datetime.datetime.strptime("2022-01-01", "%Y-%m-%d").timetuple().tm_yday -> 1

    Args:
        date_of_acquisition:

    Returns:
        (1-0.01673*cos(0.0172*(t-4)))
    """
    tm_yday = date_of_acquisition.timetuple().tm_yday # from 1 to 365 (or 366!)
    return 1 - 0.01673 * np.cos(0.0172 * (tm_yday - 4))


def observation_date_correction_factor(center_coords:Tuple[float, float], date_of_acquisition:datetime,
                                       crs_coords:Optional[str]=None,) -> float:
    """
    returns  (pi * d^2) / cos(solarzenithangle/180*pi)

    Args:
        center_coords: location being considered (x,y) (long, lat if EPSG:4326) 
        date_of_acquisition:
        crs_coords: if None it will assume center_coords are in EPSG:4326

    Returns:
        correction factor

    """
    from pysolar.solar import get_altitude
    from rasterio import warp

    if crs_coords is not None and not window_utils.compare_crs(crs_coords, "EPSG:4326"):
        centers_long, centers_lat = warp.transform(crs_coords,
                                                   {'init': 'epsg:4326'}, [center_coords[0]], [center_coords[1]])
        centers_long = centers_long[0]
        centers_lat = centers_lat[0]
    else:
        centers_long = center_coords[0]
        centers_lat = center_coords[1]
    
    # Get Solar Altitude (in degrees)
    solar_altitude = get_altitude(latitude_deg=centers_lat, longitude_deg=centers_long,
                                  when=date_of_acquisition)
    sza = 90 - solar_altitude
    d = earth_sun_distance_correction_factor(date_of_acquisition)

    return np.pi*(d**2) / np.cos(sza/180.*np.pi)


def radiance_to_reflectance(data:GeoTensor, solar_irradiance:Union[List[float], np.array],
                            date_of_acquisition:datetime) -> GeoTensor:
    """

    toaBandX = (radianceBandX / 100 * pi * d^2) / (cos(solarzenithangle/180*pi) * solarIrradianceBandX)

    ESA reference of ToA calculation:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-1c/algorithm

    where:
        d = earth_sun_distance_correction_factor(date_of_acquisition)
        solarzenithangle = is obtained from the date of aquisition and location

    Args:
        data:  (C, H, W) tensor with expected units (of AVIRIS-NG):
        microwatts per centimeter_squared per nanometer per steradian µW /(nm cm² sr)
        solar_irradiance: (C,) vector units: W/m²/nm
        date_of_acquisition: date of acquisition to compute the solar zenith angles

    Returns:
        GeoTensor with ToA on each channel

    """

    solar_irradiance = np.array(solar_irradiance)[:, np.newaxis, np.newaxis] # (C, 1, 1)
    assert len(data.shape) == 3, f"Expected 3 channels found {len(data.shape)}"
    assert data.shape[0] == len(solar_irradiance), \
        f"Different number of channels {data.shape[0]} than number of radiances {len(solar_irradiance)}"

    # Get latitude and longitude of the center of image to compute the solar angle
    center_coords = data.transform * (data.shape[-1] // 2, data.shape[-2] // 2)
    constant_factor = observation_date_correction_factor(center_coords, date_of_acquisition, crs_coords=data.crs)

    # µW /(nm cm² sr) to W/(nm m² sr)
    radiances = data.values * (10**(-6) / 1) * (1 /10**(-4))

    # data_toa = data.values / 100 * constant_factor / solar_irradiance
    data_toa = radiances * constant_factor / solar_irradiance
    mask = data == data.fill_value_default
    data_toa[mask] = data.fill_value_default

    return GeoTensor(values=data_toa, crs=data.crs, transform=data.transform,
                     fill_value_default=data.fill_value_default)


def load_srf_s2(cache:bool=True, path_override=None, drop_by_minimum = False) -> pd.DataFrame:
    """
    Loads spectral response function of Sentinel-2.
    This was obtained [here](https://sentinels.copernicus.eu/web/sentinel/user-guides/document-library/-/asset_publisher/xlslt4309D5h/content/sentinel-2a-spectral-responses;jsessionid=6D1029B0794C21DEEBA960ACFA22EB1A.jvm2?redirect=https%3A%2F%2Fsentinels.copernicus.eu%2Fweb%2Fsentinel%2Fuser-guides%2Fdocument-library%3Bjsessionid%3D6D1029B0794C21DEEBA960ACFA22EB1A.jvm2%3Fp_p_id%3D101_INSTANCE_xlslt4309D5h%26p_p_lifecycle%3D0%26p_p_state%3Dnormal%26p_p_mode%3Dview%26p_p_col_id%3Dcolumn-1%26p_p_col_count%3D1).

    Args:
        cache: cache the srf for other uses

    Returns:
        Table with SR of each of the bands in the columns

    """
    if cache:
        global SRF_S2
        if SRF_S2 is not None:
            return SRF_S2

    use_path = SRF_S2_FILE
    if path_override is not None:
        use_path = path_override

    srf_s2 = pd.read_csv(use_path)
    srf_s2 = srf_s2.set_index("SR_WL")

    # remove rows with all values zero
    any_not_cero = np.any((srf_s2 > 1e-6).values, axis=1)
    srf_s2 = srf_s2.loc[any_not_cero]

    if drop_by_minimum:
        # also drop bands to the minimal band we actually have
        srf_s2 = srf_s2.drop(list(range(411, drop_by_minimum)))

    if cache:
        SRF_S2 = srf_s2

    return srf_s2


def load_srf_wv3(cache:bool=True, path_override=None) -> pd.DataFrame:
    """
    Loads spectral response function of SWIR bands of WorldView-3

    Args:
        cache: cache the srf for other uses

    Returns:
        Table with SR of each of the bands in the columns
    """

    if cache:
        global SRF_WV3
        if SRF_WV3 is not None:
            return SRF_WV3

    use_path = SRF_WV3_FILE
    if path_override is not None:
        use_path = path_override

    srf_wv3 = pd.read_csv(use_path)
    srf_wv3 = srf_wv3.set_index("SR_WL")

    # remove rows with all values zero
    any_not_cero = np.any((srf_wv3 > 1e-6).values, axis=1)
    srf_wv3 = srf_wv3.loc[any_not_cero]

    if cache:
        SRF_WV3 = srf_wv3

    return srf_wv3

def transform_to_worldview_3(aviris:GeoData, bands_wv3:List[str],
                            resolution_dst:Optional[Union[float,Tuple[float,float]]]=10,
                            bands_nanometers_aviris:Optional[List[float]]=None,
                            fill_value_default:float=0.,
                            verbose:bool=False) -> GeoTensor:
    srf_wv3 = load_srf_wv3()
    return transform_to_srf(aviris, bands=bands_wv3,srf=srf_wv3,
                            resolution_dst=resolution_dst,
                            bands_nanometers_aviris=bands_nanometers_aviris,
                            fill_value_default=fill_value_default, verbose=verbose,
                            sigma_bands=None)


def transform_to_sentinel_2(aviris:GeoData, bands_s2:List[str],
                            resolution_dst:Optional[Union[float,Tuple[float,float]]]=10,
                            sensor:str="S2A",
                            bands_nanometers_aviris:Optional[List[float]]=None,
                            fill_value_default:float=0.,
                            verbose:bool=False) -> GeoTensor:
    srf_s2 = load_srf_s2()
    srf_s2sensor = srf_s2[[c for c in srf_s2.columns if sensor in c]].copy()
    srf_s2sensor.columns = [c.replace(f"{sensor}_SR_AV_","") for c in srf_s2sensor.columns]

    # Compute anti_aliasing_sigma per band
    resolution_or = aviris.res
    resolution_or_xy = max(resolution_or[0], resolution_or[1])

    resolution_bands = [BANDS_S2_RESOLUTION[b] for b in bands_s2]
    # (scale - 1) / 2
    sigma_bands = np.array([max((r / resolution_or_xy - 1) / 2, 0) for r in resolution_bands])

    return transform_to_srf(aviris, bands=bands_s2,srf=srf_s2sensor,
                            resolution_dst=resolution_dst,
                            bands_nanometers_aviris=bands_nanometers_aviris,
                            fill_value_default=fill_value_default, verbose=verbose,
                            sigma_bands=sigma_bands)

def transform_to_srf(aviris:GeoData, bands:List[str],
                     srf:pd.DataFrame,
                     resolution_dst:Optional[Union[float,Tuple[float,float]]]=10,
                     bands_nanometers_aviris:Optional[List[float]]=None,
                     fill_value_default:float=0.,
                     sigma_bands:Optional[np.array]=None,
                     verbose:bool=False) -> GeoTensor:

    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))

    if bands_nanometers_aviris is None:
        bands_nanometers_aviris = [float(b.replace(" Nanometers", "")) for b in aviris.descriptions]
    else:
        assert aviris.shape[0] == len(bands_nanometers_aviris), f"Different number of bands {aviris.shape[0]} and band frequency centers {len(bands_nanometers_aviris)}"

    # Construct aviris frequencies in the same resolution as srf_s2
    bands_index_aviris = np.arange(0, len(bands_nanometers_aviris))
    interp = interpolate.interp1d(bands_nanometers_aviris, bands_index_aviris, kind="nearest")
    y_nearest = interp(srf.index).astype(int)
    # if verbose:
    #     print("y_nearest", y_nearest)
    #     print("bands_nanometers_aviris", bands_nanometers_aviris)
    #     print("bands_index_aviris", bands_index_aviris)
    #     print("srf.index", srf.index)
    table_aviris_as_sr_s2 = pd.DataFrame({"SR_WL": srf.index, "AVIRIS_band": y_nearest})
    table_aviris_as_sr_s2 = table_aviris_as_sr_s2.set_index("SR_WL")

    output_array_spectral = np.full((len(bands),) + aviris.shape[-2:],
                                    fill_value=fill_value_default, dtype=np.float32)

    for i,column_name in enumerate(bands):
        if verbose:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}({i}/{len(bands)}) Processing band {column_name}")
        mask_zero = srf[column_name] <= 1e-4
        weight_per_wavelength = srf.loc[~mask_zero, [column_name]].copy()

        assert weight_per_wavelength.shape[0] >= 0, f"No weights found! {weight_per_wavelength}"

        # Join with table of previous chunk
        weight_per_wavelength = weight_per_wavelength.join(table_aviris_as_sr_s2)

        assert weight_per_wavelength.shape[0] >= 0, "No weights found!"

        # Normalize the SRF to sum one
        column_name_norm = f"{column_name}_norm"
        weight_per_wavelength[column_name_norm] = weight_per_wavelength[column_name] / weight_per_wavelength[
            column_name].sum()
        weight_per_aviris_band = weight_per_wavelength.groupby("AVIRIS_band")[[column_name_norm]].sum()

        # Load aviris bands
        if verbose:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t Loading {len(weight_per_aviris_band.index)} bands")
            # print("these ones:", weight_per_aviris_band.index)
        aviris_s2_band_i = aviris.isel({"band": weight_per_aviris_band.index}).load()
        if verbose:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t bands loaded, computing tensor")

        missing_values = np.any(
            aviris_s2_band_i.values == aviris_s2_band_i.fill_value_default,
            axis=0)

        output_array_spectral[i] = np.sum(weight_per_aviris_band[column_name_norm].values[:, np.newaxis,
                                          np.newaxis] * aviris_s2_band_i.values,
                                          axis=0)
        output_array_spectral[i][missing_values] = fill_value_default

    geotensor_spectral = GeoTensor(output_array_spectral, transform=aviris.transform,
                                   crs=aviris.crs,
                                   fill_value_default=fill_value_default)

    if (resolution_dst is None) or (resolution_dst == geotensor_spectral.res):
        return geotensor_spectral


    return read.resize(geotensor_spectral, resolution_dst=resolution_dst,
                       anti_aliasing=True, anti_aliasing_sigma=sigma_bands)


def read_aviris(aviris_img_folder_or_path:str, columns_read:int=50,
                return_windows:bool=False) -> Union[RasterioReader, List[rasterio.windows.Window]]:
    """
    returns a reader object of an AVIRIS image. This function figures out is meant to figure out if the
    image is in ENVI format or as separated tif files in the bucket.

    Optionally it returns a list of windows where reading should be optimal

    Examples:
        # Images are stored according to process_aviris.save_aviris_cog
        rdn = read_aviris("gs://starcop/Permian/data/ang20191007t175016/")

        # Image stored as a single GeoTIFF
        rdn = read_aviris("gs://starcop/Permian/images/ang20191007t175016.tif")

        # ENVI image
        _, aviris_folder = process_aviris.download_aviris("ang20191007t175016",
                                                          remove_targz_file=True,
                                                          path_untar_folder_base="/path/to/save/untar/",
                                                          display_progress_bar_download=True)

        rdn = read_aviris(aviris_folder)

    Args:
        aviris_img_folder_or_path: path to find AVIRIS image
        columns_read: optional, used only if return_windows
        return_windows: if `True` returns a list of windows where reading should be optimal

    Returns:

    """
    if aviris_img_folder_or_path.endswith(".tif"):
        rdn = RasterioReader(aviris_img_folder_or_path)
        if return_windows:
            windows = rdn.block_windows()
    else:
        if aviris_img_folder_or_path.endswith("/"):
            aviris_img_folder_or_path = aviris_img_folder_or_path[:-1]

        fsor = get_filesystem(aviris_img_folder_or_path)

        # Check if it's ENVI
        name_img = os.path.basename(aviris_img_folder_or_path)
        envi_file_path = os.path.join(aviris_img_folder_or_path, f"{name_img}_img")
        if fsor.exists(envi_file_path):
            rdn = RasterioReader(envi_file_path)
            if return_windows:
                windows = []
                for col_off in range(0, rdn.shape[1], columns_read):
                    width = min(columns_read, rdn.shape[1] - col_off)
                    wv = rasterio.windows.Window(row_off=0, col_off=col_off, height=rdn.shape[0],
                                                 width=width)
                    windows.append(wv)

        else:
            # Assume is one file per channel
            metadata_path = os.path.join(aviris_img_folder_or_path, "metadata.json")
            assert fsor.exists(metadata_path), f"Could not find AVIRIS image in {aviris_img_folder_or_path}"
            metadata = read_json_from_gcp(metadata_path)

            rdn = RasterioReader([os.path.join(aviris_img_folder_or_path, f"{i}.tif") for i in range(len(metadata["wavelengths"]))],
                                 stack=False, check=False)
            if return_windows:
                windows = rdn.block_windows()

    if return_windows:
        return rdn, windows

    return rdn