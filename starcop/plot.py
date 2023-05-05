from typing import List, Union, Optional, Dict, Any, Tuple

import matplotlib.figure
import numpy as np
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mask_to_rgb(mask:Union[Tensor,np.array], values:Union[List[int],np.array], colors_cmap:np.array) -> np.array:
    """
    Given a 2D mask it assigns to each value of the mask the corresponding color

    Args:
        mask:  array of shape (H, W) with the mask
        values: 1D list with values of the mask to assign to each color in colors_map.
        colors_cmap: 2D array of shape (len(values), 3) or (len(values), 4) with colors to assign to each value in
            values. Colors assumed to be floats in [0,1]

    Returns:
        (3, H, W) or (4, H, W) tensor with rgb(a) values of the mask.
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    mask = np.asanyarray(mask)
    assert len(values) == len(
        colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"

    assert len(mask.shape) == 2, f"Expected only 2D array found {mask.shape}"
    mask_return = np.zeros((colors_cmap.shape[1],) + mask.shape[:2], dtype=np.uint8)
    colores = np.array(np.round(colors_cmap * 255), dtype=np.uint8)
    for i, c in enumerate(colores):
        for _j in range(len(c)):
            mask_return[_j][mask == values[i]] = c[_j]
    return np.transpose(mask_return, (1,2,0))


def plot_mask_categorical(mask:Union[Tensor, np.ndarray], values:Union[List[int], np.array], colors_cmap:np.array,
                          interpretation:Optional[Union[List[str], np.array]]=None, ax:Optional[Axis]=None,
                          loc_legend:str='upper right') -> Axis:
    rgb_mask = mask_to_rgb(mask, values, colors_cmap)
    if ax is None:
        ax = plt.gca()

    ax.imshow(rgb_mask, interpolation="nearest")
    if interpretation is not None:
        patches = []
        for c, interp in zip(colors_cmap, interpretation):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches, loc=loc_legend)
    return ax

COLORS_DIFFERENCES = np.array([[0,0,0], # correct no-plume
                               [255, 0, 0], # plume missed (red)
                               [220, 220, 0], # plume overprediced (yellow)
                               [0,200,0]  # correct plume (green)
                               ]) / 255

INTERPRETATION_DIFFERENCES = ["correct no-plume", "false plume", "false no-plume", "correct plume"]

def plot_differences(differences:Union[Tensor,np.ndarray], ax:Optional[Axis]=None, legend:bool=True)->Axis:
    if legend:
        interpretation = INTERPRETATION_DIFFERENCES
    else:
        interpretation = None

    return plot_mask_categorical(differences, values=[0,1,2,3], colors_cmap=COLORS_DIFFERENCES,
                                 interpretation=interpretation,
                                 ax=ax)

def show_3_bands(tensor, ax):
    tensor = tensor.squeeze().clamp(0, 1)
    assert tensor.shape[0] == 3, f"Expected (C, H, W) tensor found {tensor.shape}"
    assert len(tensor.shape) == 3, f"Expected (C, H, W) tensor found {tensor.shape}"

    ax.imshow(np.transpose(np.asanyarray(tensor),(1,2,0)))

def show_1_band(tensor, ax, kwargs_imshow:Optional[Dict[str,Any]]=None, add_colorbar = False):
    tensor = tensor.squeeze()
    assert len(tensor.shape) == 2, f"Expected (H, W) tensor found {tensor.shape}"
    if kwargs_imshow is None:
        kwargs_imshow = {}
    im = ax.imshow(tensor,**kwargs_imshow)
    
    if add_colorbar:
        colorbar_next_to(im, ax)

def colorbar_next_to(im, ax, size='5%',pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    plt.gcf().colorbar(im, cax=cax, orientation='vertical')


PLOTTING_FUNCTIONS = {
    "rgb_aviris": {"input_products": ["TOA_AVIRIS_460nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_640nm"],
                   "tensor": "input_norm", "plot_fun": show_3_bands},
    "rgb_s2a": {"input_products": ["TOA_S2A_B4", "TOA_S2A_B3", "TOA_S2A_B2"],
                   "tensor": "input_norm", "plot_fun": show_3_bands},
    "swirnirred_s2a": {"input_products": ["TOA_S2A_B11", "TOA_S2A_B8", "TOA_S2A_B4"],
                       "tensor": "input_norm", "plot_fun": show_3_bands},
    "aviris_ratios_first": {"input_products": ['ratio_aviris_2350_2310_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "aviris_ratios_second": {"input_products": ['ratio_aviris_2350_2360_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "aviris_ratios_third": {"input_products": ['ratio_aviris_2360_2310_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    
    # varon
    "wv3_ratios_varon_b7b5": {"input_products":  ['ratio_wv3_B7_B5_varon21_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_ratios_varon_b8b5": {"input_products": ['ratio_wv3_B8_B5_varon21_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_ratios_varon_b7b6": {"input_products":  ['ratio_wv3_B7_B6_varon21_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},

    # sanchez
    "wv3_ratios_sanchez_b7b7mlr": {"input_products":  ['ratio_wv3_B7_B7MLR_SanchezGarcia22_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_ratios_sanchez_b8b8mlr": {"input_products":  ['ratio_wv3_B8_B8MLR_SanchezGarcia22_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    # sanchez
    "wv3_ratios_sanchez_b7b7mlr_v2": {"input_products":  ['ratio_wv3_B7_B7MLR_SanchezGarcia22_simplediv'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_ratios_sanchez_b8b8mlr_v2": {"input_products":  ['ratio_wv3_B8_B8MLR_SanchezGarcia22_simplediv'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},

    # learned
    "wv3_lrn_bands2band8only_60ep_512_l1": {"input_products":  ['ratio_lrn_bands2band8only_60ep_512_l1'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},

    
    # sanchez mix s2 and wv3
    "wv3_mixSanchez_b7b7mlr_fromS2_9b": {"input_products":  ['ratio_wv3_B7_B7MLR_fromS2_9bands_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_mixSanchez_b7b7mlr_fromS2_5b": {"input_products":  ['ratio_wv3_B7_B7MLR_fromS2_5bands_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_mixSanchez_b8b8mlr_fromS2_9b": {"input_products":  ['ratio_wv3_B8_B8MLR_fromS2_9bands_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_mixSanchez_b8b8mlr_fromS2_5b": {"input_products":  ['ratio_wv3_B8_B8MLR_fromS2_5bands_sum_c_out'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},

    
    # individual bands
    "s2_b1": {"input_products":  ['TOA_S2B_B1'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "s2_b2": {"input_products":  ['TOA_S2B_B1'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "s2_b3": {"input_products":  ['TOA_S2B_B1'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "s2_b4": {"input_products":  ['TOA_S2B_B1'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},

    
    # individual bands
    "wv3_b1": {"input_products":  ['TOA_WV3_SWIR1'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b2": {"input_products":  ['TOA_WV3_SWIR2'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b3": {"input_products":  ['TOA_WV3_SWIR3'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b4": {"input_products":  ['TOA_WV3_SWIR4'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b5": {"input_products":  ['TOA_WV3_SWIR5'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b6": {"input_products":  ['TOA_WV3_SWIR6'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b7": {"input_products":  ['TOA_WV3_SWIR7'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},
    "wv3_b8": {"input_products":  ['TOA_WV3_SWIR8'], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, add_colorbar=True)},

    "mag1c": {"input_products": ["mag1c"], "tensor": "input_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, kwargs_imshow={"vmin":0, "vmax": 2})},
    "label": {"tensor": "output_norm",
              "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, kwargs_imshow={"vmin":0, "vmax": 1, "interpolation":"nearest"})},
    "pred": {"tensor": "prediction",
             "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, kwargs_imshow={"vmin":0, "vmax": 1})},
    "pred_binary": {"tensor": "prediction",
                    "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, kwargs_imshow={"vmin":0, "vmax": 1, "interpolation":"nearest"})},
    "weight_loss": {"tensor": "weight_loss",
                    "plot_fun": lambda tensor, ax: show_1_band(tensor, ax, kwargs_imshow={"vmin":0, "vmax": 1})},
    "differences": {"tensor": "differences",
                    "plot_fun": lambda tensor, ax: plot_differences(tensor.squeeze(), ax)},
}

@torch.no_grad()
def plot_batch(batch_with_preds:Dict[str, Tensor],
               input_products:List[str],
               products_plot:List[str], figsize_ax:Tuple[int,int]=(2,2), 
               add_id_to_title:bool=False) -> Union[matplotlib.figure.Figure, Axis]:
    batch_size = len(batch_with_preds["input"])

    fig, ax = plt.subplots(batch_size, len(products_plot), 
                           figsize=(figsize_ax[0] * len(products_plot), figsize_ax[1]*batch_size),
                           tight_layout=True, squeeze=False)

    for idx_product_plot, p in enumerate(products_plot):
        if p not in PLOTTING_FUNCTIONS:
            if p not in batch_with_preds:
                assert p in input_products, f"{p} not registered in {PLOTTING_FUNCTIONS.keys()} and not in {input_products}"
                idx_p = input_products.index(p)
                tensor_key = "input_norm"
                tensor = batch_with_preds[tensor_key][:, idx_p]
            else:
                tensor = batch_with_preds[p]

            plotting_fun = show_1_band
        else:
            if p not in batch_with_preds:
                # note, we may not have "input_products" at all!
                if "input_products" in PLOTTING_FUNCTIONS[p]:
                    input_products_idx = PLOTTING_FUNCTIONS[p]["input_products"]
                else:
                    input_products_idx = []

                    # See if input_products_idx are in batch_with_preds
                if  (len(input_products_idx) > 0) and all(ip in batch_with_preds for ip in input_products_idx):
                    if len(input_products_idx) > 1:
                        tensor = torch.cat([batch_with_preds[ip] for ip in input_products_idx], dim=0)
                    else:
                        tensor = batch_with_preds[input_products_idx[0]]
                        if p == "mag1c":
                            tensor = tensor / 1750
                else:
                    # Check if input_products_idx are in tensor
                    tensor_key = PLOTTING_FUNCTIONS[p]["tensor"]
                    assert tensor_key in batch_with_preds, f"Batch does not have keys: {p} {tensor_key}. Keys in batch: {batch_with_preds.keys()}"
                    tensor = batch_with_preds[tensor_key]
                    if tensor_key.startswith("input"):
                        idx_show = [idx for idx, input_product in enumerate(input_products) if
                                    input_product in PLOTTING_FUNCTIONS[p]["input_products"]]
                        assert len(PLOTTING_FUNCTIONS[p]["input_products"]) == len(
                            idx_show), f"Unexpected number of products"
                        tensor = tensor[:, tuple(idx_show), ...]
            else:
                tensor_key = p
                tensor = batch_with_preds[tensor_key]
                if p == "mag1c":
                    tensor = tensor / 1750

            plotting_fun = PLOTTING_FUNCTIONS[p]["plot_fun"]

        for idx_batch in range(batch_size):
            plotting_fun(tensor[idx_batch], ax[idx_batch, idx_product_plot])
            if add_id_to_title:
                ax[idx_batch, idx_product_plot].set_title(f"{p} {batch_with_preds['id'][idx_batch]}")
            elif idx_batch == 0:
                ax[idx_batch, idx_product_plot].set_title(p)
            #print()

    return fig, ax






