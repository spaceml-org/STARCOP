import warnings

import numpy as np
import torch
import torch.nn

BAND_NORMALIZATION = {
 'TOA_S2A_B1': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B10': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B11': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B12': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B2': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B3': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B4': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B5': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B6': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B7': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B8': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B8A': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2A_B9': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B1': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B10': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B11': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B12': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B2': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B3': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B4': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B5': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B6': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B7': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B8': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B8A': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_S2B_B9': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR1': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR2': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR3': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR4': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR5': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR6': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR7': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_WV3_SWIR8': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_AVIRIS_550nm': {'offset': 0, 'factor': 60, 'clip': (0, 2)},
 'TOA_AVIRIS_640nm': {'offset': 0, 'factor': 60, 'clip': (0, 2)},
 'TOA_AVIRIS_460nm': {'offset': 0, 'factor': 60, 'clip': (0, 2)},
 'TOA_AVIRIS_2004nm': {'offset': 0, 'factor': 1, 'clip': (0, 2)},
 'TOA_AVIRIS_2109nm': {'offset': 0, 'factor': 5, 'clip': (0, 2)},
 'TOA_AVIRIS_2310nm': {'offset': 0, 'factor': 4, 'clip': (0, 2)},
 'TOA_AVIRIS_2350nm': {'offset': 0, 'factor': 3, 'clip': (0, 2)},
 'TOA_AVIRIS_2360nm': {'offset': 0, 'factor': 3, 'clip': (0, 2)},
 'mag1c': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},

 'ratio_aviris_2350_2310_out':  {'offset': 0, 'factor': 0.0625, 'clip': (-2., 2.)}, # 1/16 = 0.0625
 'ratio_aviris_2350_2360_out': {'offset': 0, 'factor': 0.0625, 'clip': (-2., 2.)},
 'ratio_aviris_2360_2310_out':  {'offset': 0, 'factor': 0.0625, 'clip': (-2., 2.)}, # note: we are being gentle, we could do 1/18 instead?
    
 'ratio_wv3_B7_B5_varon21_sum_c_out': {'offset': 0, 'factor': 0.04, 'clip': (-2., 2.)}, # 1/25 = 0.04
 'ratio_wv3_B8_B5_varon21_sum_c_out': {'offset': 0, 'factor': 0.1, 'clip': (-2., 2.)},
 'ratio_wv3_B7_B6_varon21_sum_c_out': {'offset': 0, 'factor': 0.1, 'clip': (-2., 2.)},

 'ratio_wv3_B7_B7MLR_SanchezGarcia22_sum_c_out': {'offset': 0, 'factor': 0.025, 'clip': (-2., 2.)}, # 1/40=0.025
 'ratio_wv3_B8_B8MLR_SanchezGarcia22_sum_c_out': {'offset': 0, 'factor': 0.0769, 'clip': (-2., 2.)}, # 1/13=0.07692307692
    
 'ratio_wv3_B7_B7MLR_SanchezGarcia22_simplediv': {'offset': 0, 'factor': 1, 'clip': (-2., 2.)}, #
 'ratio_wv3_B8_B8MLR_SanchezGarcia22_simplediv': {'offset': -0.5, 'factor': 1, 'clip': (-2., 2.)}, # 1/15=0.0666

    
 'ratio_lrn_bands2band8only_60ep_512_l1': {'offset': 0, 'factor': 0.5, 'clip': (-2., 2.)}, # orig has useful in cca -0.5,0.5, so *2 and then clip -2,2

 'ratio_wv3_B7_B7MLR_fromS2_9bands_sum_c_out': {'offset': 0, 'factor': 1, 'clip': (-2., 2.)}, # keep
 'ratio_wv3_B7_B7MLR_fromS2_5bands_sum_c_out': {'offset': 0, 'factor': 0.1111111, 'clip': (-2., 2.)}, # *9
 'ratio_wv3_B8_B8MLR_fromS2_9bands_sum_c_out': {'offset': 0, 'factor': 0.125, 'clip': (-2., 2.)}, # *8
 'ratio_wv3_B8_B8MLR_fromS2_5bands_sum_c_out': {'offset': 0, 'factor': 0.1666666, 'clip': (-2., 2.)}, # *6

}

#torch.clamp((x-offset) / self.factor, self.clip_min_input ,self.clip_max_input)

class DataNormalizer(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings_dataset = settings.dataset
        offsets_input = []
        factors_input = []
        clip_min_input = []
        clip_max_input = []
        for p in self.settings_dataset.input_products:
            if p not in BAND_NORMALIZATION:
                warnings.warn(f"Feature {p} does not have band normalization attributes. "
                              f"It will not be normalized BUT it will be clipped to [-10, 10]")
                offsets_input.append(0)
                factors_input.append(1)
                clip_min_input.append(-10)
                clip_max_input.append(10)
            else:
                offsets_input.append(BAND_NORMALIZATION[p]["offset"])
                factors_input.append(BAND_NORMALIZATION[p]["factor"])
                clip_min_input.append(BAND_NORMALIZATION[p]["clip"][0])
                clip_max_input.append(BAND_NORMALIZATION[p]["clip"][1])

        self.offsets_input = torch.nn.Parameter(torch.from_numpy(np.array(offsets_input)[:,None,None]),
                                                requires_grad=False) # (C, 1, 1)
        self.factors_input = torch.nn.Parameter(torch.from_numpy(np.array(factors_input)[:,None,None]),
                                                requires_grad=False)
        self.clip_min_input = torch.nn.Parameter(torch.from_numpy(np.array(clip_min_input)[:,None,None]),
                                                 requires_grad=False)
        self.clip_max_input = torch.nn.Parameter(torch.from_numpy(np.array(clip_max_input)[:, None, None]),
                                                 requires_grad=False)

        offsets_output = []
        factors_output = []
        clip_min_output = []
        clip_max_output = []
        for p in self.settings_dataset.output_products:
            if p in BAND_NORMALIZATION:
                offsets_output.append(BAND_NORMALIZATION[p]["offset"])
                factors_output.append(BAND_NORMALIZATION[p]["factor"])
                clip_min_output.append(BAND_NORMALIZATION[p]["clip"][0])
                clip_max_output.append(BAND_NORMALIZATION[p]["clip"][1])

        if len(factors_output) > 0:
            assert len(factors_output) == len(self.settings_dataset.output_products), f"Some output products don't have normalization. CHECK!"
            self.factors_output = torch.nn.Parameter(torch.from_numpy(np.array(factors_output)[:,None,None]),
                                                     requires_grad=False)
            self.offsets_output = torch.nn.Parameter(torch.from_numpy(np.array(offsets_output)[:, None, None]),
                                                     requires_grad=False)
            self.clip_min_output = torch.nn.Parameter(torch.from_numpy(np.array(clip_min_output)[:, None, None]),
                                                     requires_grad=False)
            self.clip_max_output = torch.nn.Parameter(torch.from_numpy(np.array(clip_max_output)[:, None, None]),
                                                     requires_grad=False)
        else:
            self.factors_output = None
            self.offsets_output = None

    def normalize_x(self, x):
        return torch.clamp((x-self.offsets_input) / self.factors_input, self.clip_min_input ,self.clip_max_input).float()

    def denormalize_x(self, x):
        return (x * self.factors_input) + self.offsets_input

    def normalize_y(self, y):
        if self.factors_output is not None:
            return torch.clamp((y - self.offsets_output) / self.factors_output,
                               self.clip_min_output, self.clip_max_output)
        return y

    def denormalize_y(self, y):
        if self.factors_output is not None:
            return (y * self.factors_output) + self.offsets_output
        return y
