import numpy as np
from typing import List
import os
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

WV3_BANDS = [f"TOA_WV3_SWIR{w+1}"for w in range(8)]
S2_BANDS = ["B1", "B2",
            "B3", "B4",
            "B5", "B6",
            "B7", "B8",
            "B8A", "B9",
            "B10", "B11", "B12"]

S2A_BANDS = [f"TOA_S2A_{b}" for b in S2_BANDS]
S2B_BANDS = [f"TOA_S2B_{b}" for b in S2_BANDS]

AVIRIS_WAVELENGTHS = [376, 381, 386, 391, 396, 401, 406, 412, 417, 422, 427, 432, 437, 442, 447, 452, 457, 462, 467, 472, 477, 482, 487, 492, 497, 502, 507, 512, 517, 522, 527, 532, 537, 542, 547, 552, 557, 562, 567, 572, 577, 582, 587, 592, 597, 602, 607, 612, 617, 622, 627, 632, 637, 642, 647, 652, 657, 662, 667, 672, 677, 682, 687, 692, 697, 702, 707, 712, 717, 722, 727, 732, 737, 742, 747, 752, 757, 762, 767, 772, 777, 782, 787, 792, 797, 802, 807, 812, 817, 822, 827, 832, 837, 842, 847, 852, 857, 862, 867, 872, 877, 882, 887, 892, 897, 902, 907, 912, 917, 922, 927, 932, 937, 942, 947, 952, 957, 962, 967, 972, 977, 982, 988, 993, 998, 1003, 1008, 1013, 1018, 1023, 1028, 1033, 1038, 1043, 1048, 1053, 1058, 1063, 1068, 1073, 1078, 1083, 1088, 1093, 1098, 1103, 1108, 1113, 1118, 1123, 1128, 1133, 1138, 1143, 1148, 1153, 1158, 1163, 1168, 1173, 1178, 1183, 1188, 1193, 1198, 1203, 1208, 1213, 1218, 1223, 1228, 1233, 1238, 1243, 1248, 1253, 1258, 1263, 1268, 1273, 1278, 1283, 1288, 1293, 1298, 1303, 1308, 1313, 1318, 1323, 1328, 1333, 1338, 1343, 1348, 1353, 1358, 1363, 1368, 1373, 1378, 1383, 1388, 1393, 1398, 1403, 1408, 1413, 1418, 1423, 1428, 1433, 1438, 1443, 1448, 1453, 1458, 1463, 1468, 1473, 1478, 1483, 1488, 1493, 1498, 1503, 1508, 1513, 1518, 1523, 1528, 1533, 1538, 1543, 1548, 1553, 1558, 1563, 1568, 1574, 1579, 1584, 1589, 1594, 1599, 1604, 1609, 1614, 1619, 1624, 1629, 1634, 1639, 1644, 1649, 1654, 1659, 1664, 1669, 1674, 1679, 1684, 1689, 1694, 1699, 1704, 1709, 1714, 1719, 1724, 1729, 1734, 1739, 1744, 1749, 1754, 1759, 1764, 1769, 1774, 1779, 1784, 1789, 1794, 1799, 1804, 1809, 1814, 1819, 1824, 1829, 1834, 1839, 1844, 1849, 1854, 1859, 1864, 1869, 1874, 1879, 1884, 1889, 1894, 1899, 1904, 1909, 1914, 1919, 1924, 1929, 1934, 1939, 1944, 1949, 1954, 1959, 1964, 1969, 1974, 1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019, 2024, 2029, 2034, 2039, 2044, 2049, 2054, 2059, 2064, 2069, 2074, 2079, 2084, 2089, 2094, 2099, 2104, 2109, 2114, 2119, 2124, 2129, 2134, 2139, 2144, 2150, 2155, 2160, 2165, 2170, 2175, 2180, 2185, 2190, 2195, 2200, 2205, 2210, 2215, 2220, 2225, 2230, 2235, 2240, 2245, 2250, 2255, 2260, 2265, 2270, 2275, 2280, 2285, 2290, 2295, 2300, 2305, 2310, 2315, 2320, 2325, 2330, 2335, 2340, 2345, 2350, 2355, 2360, 2365, 2370, 2375, 2380, 2385, 2390, 2395, 2400, 2405, 2410, 2415, 2420, 2425, 2430, 2435, 2440, 2445, 2450, 2455, 2460, 2465, 2470, 2475, 2480, 2485, 2490, 2495, 2500]
def raw_bands_available() -> List[str]:
    bands = []
    for wv in AVIRIS_WAVELENGTHS + [550, 640, 460]:
        bands.append(f"TOA_AVIRIS_{wv}nm")
    bands.extend(WV3_BANDS)
    bands.extend(S2A_BANDS+S2B_BANDS)
    bands.append('mag1c')
    bands.append('labelbinary')
    bands.append('label_rgba')
    return bands

def weight_mag1c(mag1c:np.array) -> np.array:
    mag1c_positive = np.clip(mag1c / 400, 0.1, 1)
    # return labelbinary * mag1c_positive  + (1-labelbinary) * mag1c_positive
    return mag1c_positive

def no_outliers(d, percentile=5):
    upper_quartile = np.percentile(d, 100-percentile)
    lower_quartile = np.percentile(d, percentile)
    return d[np.where((d >= lower_quartile) & (d <= upper_quartile))]

def ratio_2c_match_c_from_sums_outlier(background_channel:np.array, signal:np.array, p=5, zero_value_out=-.6) -> np.array: # usually called as ratio_2_match_c_from_sums(band_signal, band_bg)
    signal_flat = signal.flatten()
    background_flat = background_channel.flatten()
    
    zero_signal_and_background = (signal < 1e-6) & (background_channel < 1e-6)
    
    background_sum = np.sum(no_outliers(background_flat, p))
    signal_sum = np.sum(no_outliers(signal_flat, p))
    
    c = (background_sum / signal_sum)
    R = (c * signal - background_channel) / (background_channel + 1e-6)
    
    R[zero_signal_and_background] = zero_value_out
    
    return R

def ratio_MLR_local(bands_bg, band_target_signal, division="c_matched_outliers", autoclip=False):
    # multiple linear regression (MLR) of B1–B6 with B7
    # method [Sanchez-Garcia et al.]

    # division simple (L / L_ref) or residual/c_matched_outliers (L-L_ref / L_ref) has different ranges
    # simple ends up around 1., residual around 0.
    
    reshape_shape = band_target_signal.shape
    band_target_signal_flat = band_target_signal.flatten()
    bands_bg = [b.flatten() for b in bands_bg]
    bands_bg = np.asarray(bands_bg)
    bands_bg = np.swapaxes(bands_bg, 0,1)
    # MLE
    model = LinearRegression()
    model.fit(bands_bg,band_target_signal_flat)
    # print("Coeficients and intercept:", model.coef_, model.intercept_)
    # "predict" on the same data as in train
    predictions = model.predict(bands_bg)
    band_mle_reconstruction = predictions.reshape((reshape_shape))

    if division=="simple":
        # original paper states: L / L_ref (where L_ref is the one obtained from MLE)
        R = band_target_signal / (band_mle_reconstruction + 1e-6)
        R = np.where(band_target_signal == 0.0, 1.0, R) # set to "zero" where we had no data

    if division=="simple_plus":
        # original paper states: L / L_ref (where L_ref is the one obtained from MLE)
        R = band_target_signal / (band_mle_reconstruction + 1e-6)
        R = 0.0 - R
        # then impose local norm within each tile (assuming this is the 512x512)
        R = ( R - np.mean(R) ) / np.std(R)

        R = np.where(band_target_signal == 0.0, np.min(R), R) # set to "zero" where we had no data
        
    elif division=="residual":
        # we can assume the band_target_signal contains signal, while band_mle_reconstruction is the background
        R = (band_target_signal - band_mle_reconstruction) / (band_mle_reconstruction + 1e-6)
        
        R = np.where(band_target_signal == 0.0, 0.0, R) # set to "zero" where we had no data
    elif division=="c_matched_outliers":
        # we can assume the band_target_signal contains signal, while band_mle_reconstruction is the background
        # R = (band_target_signal - band_mle_reconstruction) / (band_mle_reconstruction + 1e-6)

        zero_value_out=-.5
        R = ratio_2c_match_c_from_sums_outlier(
            band_target_signal,band_mle_reconstruction,
            zero_value_out=zero_value_out) # bg, signal this time (will get flipped to -R)
        R = np.where(band_target_signal == 0.0, zero_value_out, R) # set to "zero" where we had no data
    else: assert False
    if autoclip: # when running on the large 512x512 tiles this is kinda needed!
        R = np.clip(R, -0.2, +0.2)
    return R

def ratio_MLR_local_5IN(IN1,IN2,IN3,IN4,IN5,target_B, division="c_matched_outliers", autoclip=False):
    bands_bg = [IN1,IN2,IN3,IN4,IN5]
    band_target_signal = target_B
    return ratio_MLR_local(bands_bg, band_target_signal, division=division, autoclip=autoclip)

def ratio_MLR_local_9IN(IN1,IN2,IN3,IN4,IN5,IN6,IN7,IN8,IN9,target_B, division="c_matched_outliers", autoclip=False):
    bands_bg = [IN1,IN2,IN3,IN4,IN5,IN6,IN7,IN8,IN9]
    band_target_signal = target_B
    return ratio_MLR_local(bands_bg, band_target_signal, division=division, autoclip=autoclip)

def ratio_MLR_local_5IN_simplediv(IN1,IN2,IN3,IN4,IN5,target_B, division="simple_plus", autoclip=False):
    bands_bg = [IN1,IN2,IN3,IN4,IN5]
    band_target_signal = target_B
    return ratio_MLR_local(bands_bg, band_target_signal, division=division, autoclip=autoclip)

        
model_preloaded = None
def use_pretrained_model_b1to6_b8(inB1,inB2,inB3,inB4,inB5,inB6,outB8, 
                         experiment_path="wv3_cnn_v2_bands2band8only_60ep_512_l1/2022-10-21_14-08"):
    # Experimental feature!
    global model_preloaded
    import torch
    if model_preloaded is None:
        print("model not instantiated, will load!")
        import omegaconf
        
        config = omegaconf.OmegaConf.load("../scripts/configs/config.yaml")
        import fsspec
        fs = fsspec.filesystem("gs")
        with fs.open(f"gs://starcop/experiments/{experiment_path}/.hydra/config.yaml", "r") as f:
            config_model = omegaconf.OmegaConf.load(f)

        config = omegaconf.OmegaConf.merge(config, config_model)
        config.dataloader.batch_size = 1
        
        from starcop.models.model_module_regression import ModelModuleRegression
        checkpoint_path = "gs://starcop/experiments/"+experiment_path+"/final_checkpoint_model.ckpt"
        model = ModelModuleRegression.load_from_checkpoint(checkpoint_path, settings=config)
        # to prevent reloading each time!
        model_preloaded = model

    inp = np.asarray([inB1[0], inB2[0], inB3[0], inB4[0], inB5[0], inB6[0]])
    inp = np.asarray([inp])
    target = np.asarray([outB8])

    inp = torch.from_numpy(inp)
    target = torch.from_numpy(target)
    
    inp = inp.to(model_preloaded.device, dtype=torch.float)
    target = target.to(model_preloaded.device, dtype=torch.float)
    
    # print("inp",inp.shape)
    output=model_preloaded(inp)
    # print("output,target",output.shape, target.shape)
    
    output = output.detach().cpu().numpy()[0][0]
    target = target.cpu().numpy()[0][0]
    # print(output.shape, target.shape)
    
    zero_value_out = -0.5
    R = ratio_2c_match_c_from_sums_outlier(
        target,output,
        zero_value_out=zero_value_out)
    R = np.where(target == 0.0, zero_value_out, R) # set to "zero" where we had no data
    return R



def lr_bands(regressors:List[np.array], signal:np.array) -> np.array:
    """ Test method in Sanchez-Garcia et al 2021 """
    regressors_flat = [r.flatten() for r in regressors]
    signal_flat = signal.flatten()

    regressors_flat_ones = np.stack(regressors_flat + [np.ones(signal_flat.shape[0])],
                                    axis=-1)
    c2,_,_,_ = np.linalg.lstsq(regressors_flat_ones, signal_flat)
    prediction = np.stack(regressors+[np.ones(signal.shape)], axis=-1) @ c2
    prediction = prediction[..., 0]
    residuals = (prediction - signal) # / (ch11 + 1e-6)
    return residuals


FEATURES = {
    "weight_mag1c" : {"function": weight_mag1c, "inputs": ["mag1c"], "fill_value_default":None},
    # AVIRIS bands ratios:
    "ratio_aviris_2350_2310_out" : {"function": ratio_2c_match_c_from_sums_outlier, "inputs": ["TOA_AVIRIS_2350nm", "TOA_AVIRIS_2310nm"], "fill_value_default":None},
    "ratio_aviris_2350_2360_out" : {"function": ratio_2c_match_c_from_sums_outlier, "inputs": ["TOA_AVIRIS_2350nm", "TOA_AVIRIS_2360nm"], "fill_value_default":None},
    "ratio_aviris_2360_2310_out" : {"function": ratio_2c_match_c_from_sums_outlier, "inputs": ["TOA_AVIRIS_2360nm", "TOA_AVIRIS_2310nm"], "fill_value_default":None},
    # WV3 bands ratios:
    # sum matched c ratios between bands, division follows Varon21
    "ratio_wv3_B7_B5_varon21_sum_c_out" : {"function": ratio_2c_match_c_from_sums_outlier, "inputs": ["TOA_WV3_SWIR7", "TOA_WV3_SWIR5"], "fill_value_default":None},
    "ratio_wv3_B8_B5_varon21_sum_c_out" : {"function": ratio_2c_match_c_from_sums_outlier, "inputs": ["TOA_WV3_SWIR8", "TOA_WV3_SWIR5"], "fill_value_default":None},
    "ratio_wv3_B7_B6_varon21_sum_c_out" : {"function": ratio_2c_match_c_from_sums_outlier, "inputs": ["TOA_WV3_SWIR7", "TOA_WV3_SWIR6"], "fill_value_default":None},
    # multiple linear regression, simple division or similar as before, follows SanchezGarcia22
    "ratio_wv3_B7_B7MLR_SanchezGarcia22_sum_c_out" : {"function": ratio_MLR_local_5IN, 
            "inputs": ["TOA_WV3_SWIR1", "TOA_WV3_SWIR2", "TOA_WV3_SWIR4", "TOA_WV3_SWIR5", "TOA_WV3_SWIR6", 
                       "TOA_WV3_SWIR7"], "fill_value_default":None},
    "ratio_wv3_B8_B8MLR_SanchezGarcia22_sum_c_out" : {"function": ratio_MLR_local_5IN, 
            "inputs": ["TOA_WV3_SWIR1", "TOA_WV3_SWIR2", "TOA_WV3_SWIR4", "TOA_WV3_SWIR5", "TOA_WV3_SWIR6", 
                       "TOA_WV3_SWIR8"], "fill_value_default":None},


    "ratio_wv3_B7_B7MLR_SanchezGarcia22_simplediv" : {"function": ratio_MLR_local_5IN_simplediv, 
            "inputs": ["TOA_WV3_SWIR1", "TOA_WV3_SWIR2", "TOA_WV3_SWIR4", "TOA_WV3_SWIR5", "TOA_WV3_SWIR6", 
                       "TOA_WV3_SWIR7"], "fill_value_default":None},
    "ratio_wv3_B8_B8MLR_SanchezGarcia22_simplediv" : {"function": ratio_MLR_local_5IN_simplediv, 
            "inputs": ["TOA_WV3_SWIR1", "TOA_WV3_SWIR2", "TOA_WV3_SWIR4", "TOA_WV3_SWIR5", "TOA_WV3_SWIR6", 
                       "TOA_WV3_SWIR8"], "fill_value_default":None},

    
    # Learned models:
    "ratio_lrn_bands2band8only_60ep_512_l1" : {"function": use_pretrained_model_b1to6_b8, 
            "inputs": ["TOA_WV3_SWIR1", "TOA_WV3_SWIR2", "TOA_WV3_SWIR3", "TOA_WV3_SWIR4", "TOA_WV3_SWIR5", "TOA_WV3_SWIR6", "TOA_WV3_SWIR8"], "fill_value_default":None},

    # Sanchez ratios, but if we had S2 sensor with one WV3 band
    # S2: b2, b3, b4, b5, b6, b7, b8, b8a, b11 ~ use only some of these ...
    # WV3: B7 or B8
    "ratio_wv3_B7_B7MLR_fromS2_9bands_sum_c_out" : {"function": ratio_MLR_local_9IN, 
            "inputs": ["TOA_S2B_B2", "TOA_S2B_B3", "TOA_S2B_B4", "TOA_S2B_B5", "TOA_S2B_B6", "TOA_S2B_B7", "TOA_S2B_B8", "TOA_S2B_B8A", "TOA_S2B_B11",
                       "TOA_WV3_SWIR7"], "fill_value_default":None},

    "ratio_wv3_B7_B7MLR_fromS2_5bands_sum_c_out" : {"function": ratio_MLR_local_5IN, 
            "inputs": ["TOA_S2B_B2", "TOA_S2B_B3", "TOA_S2B_B4", "TOA_S2B_B8", "TOA_S2B_B11",
                       "TOA_WV3_SWIR7"], "fill_value_default":None},

    "ratio_wv3_B8_B8MLR_fromS2_9bands_sum_c_out" : {"function": ratio_MLR_local_9IN, 
            "inputs": ["TOA_S2B_B2", "TOA_S2B_B3", "TOA_S2B_B4", "TOA_S2B_B5", "TOA_S2B_B6", "TOA_S2B_B7", "TOA_S2B_B8", "TOA_S2B_B8A", "TOA_S2B_B11",
                       "TOA_WV3_SWIR8"], "fill_value_default":None},

    "ratio_wv3_B8_B8MLR_fromS2_5bands_sum_c_out" : {"function": ratio_MLR_local_5IN, 
            "inputs": ["TOA_S2B_B2", "TOA_S2B_B3", "TOA_S2B_B4", "TOA_S2B_B8", "TOA_S2B_B11",
                       "TOA_WV3_SWIR8"], "fill_value_default":None},

    
    
}


def extract_features(features:List[str], dataframe):
    features_inputs = set()
    exists_all = {}
    for f in features:
        exists_all[f] = dataframe["folder"].apply(lambda x: os.path.exists(os.path.join(x, f"{f}.tif"))).all()
        if not exists_all[f]:
            print(f"Feature {f} does not exists. It will be generated")
            inputs = FEATURES[f]["inputs"]
            features_inputs = features_inputs.union(set(inputs))

    if all(exists_all[f] for f in features):
        return

    from georeader.geotensor import GeoTensor
    from georeader.save_cog import save_cog
    from . import sampling_dataset

    dataset = sampling_dataset.WindowDataset(dataframe=dataframe.copy(),
                                             products=list(features_inputs),
                                             normalize_by_acquisition_date=False, proposed_mask=False,
                                             read_rgb_path=False, read_label_path=False, output_size=None)

    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        input_tensors = dataset[idx]
        folder = dataset.dataframe.iloc[idx].folder
        for f in features:
            if exists_all[f]:
                continue
            product_path = os.path.join(folder, f"{f}.tif")
            if os.path.exists(product_path):
                continue

            nparray = FEATURES[f]["function"](*[input_tensors[iname].values for iname in FEATURES[f]["inputs"]])
            it0 = input_tensors[FEATURES[f]["inputs"][0]]
            save_cog(GeoTensor(nparray, transform=it0.transform, crs=it0.crs, fill_value_default=None),
                     path_tiff_save=product_path, descriptions=[f],
                     profile={"BLOCKSIZE": 128})

