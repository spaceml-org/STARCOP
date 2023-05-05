import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from starcop.models.model_module import ModelModule, differences, pred_classification
import torch
import torchmetrics
import starcop.metrics as starcopmetrics
import starcop.plot as starcoplot
from typing import Optional, List, Dict, Tuple
from numbers import Number
import tempfile
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from starcop.torch_utils import to_device
import pytorch_lightning as pl
from omegaconf import OmegaConf
from kornia.morphology import dilation as kornia_dilation
from kornia.morphology import erosion as kornia_erosion
from starcop.data.normalizer_module import DataNormalizer

@torch.no_grad()
def run_validation(model:ModelModule, dataloader:DataLoader,
                   products_plot:Optional[List[str]]=None, verbose:bool=True,
                   thresholds:Optional[np.array]=None,
                   show_plots:bool=True, path_save_results:Optional[str]=None,
                   skip_saving_plots = False, # for faster runs
                   mask_from_magic = False, # in some cases we may want to mask out predictions (not with the main AVIRIS dataset, but with orthorectified EMIT samples)
                  ) -> Tuple[pd.DataFrame, Dict[str, Number]]:

    assert dataloader.batch_size == 1, "This function is expected to run with batch_size 1"


    if thresholds is None:
        thresholds = [0,1e-3,1e-2]+ np.arange(0.5,.96,.05).tolist() + [.99,.995,.999]

    # Sort thresholds from high to low
    thresholds = np.sort(thresholds)
    thresholds = thresholds[-1::-1]
    confusion_metric_thresholds = []
    for thr in thresholds:
        cm_thr = torchmetrics.ConfusionMatrix(num_classes=2)
        cm_thr.to(model.device)
        confusion_metric_thresholds.append({"confusion_matrix": cm_thr,
                                            "threshold": thr})


    # Add products to plot that are not in the dataset for plotting
    if products_plot is None:
        products_plot = []
    else:
        pplot_add = []
        for pp in products_plot:
            if pp == "mag1c":
                if "mag1c" not in dataloader.dataset.input_products:
                    print(f"Adding mag1c as extra products to dataset for plotting purposes")
                    dataloader.dataset.add_extra_products(["mag1c"])
            elif pp == "rgb_aviris":
                if not all(b in dataloader.dataset.input_products for b in ["TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]):
                    print(f"RGB aviris not in input. Adding to dataset for plotting purposes")
                    dataloader.dataset.add_rgb_aviris = True
                

    confusion_metric = torchmetrics.ConfusionMatrix(num_classes=2)
    model.eval()
    
    confusion_metric.to(model.device)

    if path_save_results.startswith("gs://"):
        path_save_results_temp = tempfile.mkdtemp(prefix="starcop")
        path_save_results_remote = path_save_results
        path_save_results = path_save_results_temp
    else:
        path_save_results_remote = None

    out_data = []
    for idx, plume_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        plume_data = model.batch_with_preds(to_device(plume_data, model.device))
        
        if mask_from_magic:
            # Special case, likely only when we are working with orthorectified EMIT data - there we need to ignore parts of the image that include area outside
            # Note: Masking may not be needed - as long as we don't really care for the number of TNs
            assert "nodata_mask" in plume_data.keys() # the nan mask has to be provided from the dataloader
            nan_mask = plume_data["nodata_mask"][0]
            y_flat = plume_data["output_norm"].cpu().numpy().flatten()
            pred_binary_flat = plume_data["pred_binary"].cpu().numpy().flatten()
            nan_mask_flat = nan_mask.cpu().numpy().flatten()

            # not very pretty, but torchmetrics.ConfusionMatrix is weird, would rather do this outside like this ...
            tmp_y = []
            tmp_pred_binary = []
            for i in range(len(y_flat)):
                if nan_mask_flat[i] == 0: # valid values
                    tmp_y.append(y_flat[i])
                    tmp_pred_binary.append(pred_binary_flat[i])
        
            tmp_y = torch.from_numpy(np.asarray(tmp_y)).long().to(model.device)
            tmp_pred_binary = torch.from_numpy(np.asarray(tmp_pred_binary)).long().to(model.device)
            
            cm_iter = confusion_metric(tmp_pred_binary, tmp_y).cpu()

            # the other func can still use this ...
            y_long = plume_data["output_norm"].long()

            
        else:
            y_long = plume_data["output_norm"].long()
            cm_iter = confusion_metric(plume_data["pred_binary"], y_long).cpu()

        metrics_iter = {}
        for fun in starcopmetrics.METRICS_CONFUSION_MATRIX + [starcopmetrics.TP, starcopmetrics.TN, starcopmetrics.FP, starcopmetrics.FN]:
            metrics_iter[fun.__name__] = fun(cm_iter).item()
            
        # PR curve
        for cm_thres in confusion_metric_thresholds: # thresholds from high to low
            cm_iter_thres = cm_thres["confusion_matrix"]
            if hasattr(model, "apply_threshold"):
                pred_binary = model.apply_threshold(plume_data["prediction"], cm_thres["threshold"])
            else:
                pred_binary = (plume_data["prediction"] > cm_thres["threshold"]).long()
            
            cm_iter_thres.update(pred_binary, y_long)
        
        # Save metrics
        plume_data = to_device(plume_data, torch.device("cpu"))

        metrics_iter["id"] = plume_data["id"][0]
        metrics_iter["label_pixels_plume"] = y_long[0, 0].cpu().sum().item()
        metrics_iter["has_plume"] = plume_data['has_plume'][0].item()
        metrics_iter["pred_classification"] = plume_data['pred_classification'][0, 0].item()
        metrics_iter["pred_pixels_plume"] = plume_data['pred_binary'][0, 0].sum().item()
        out_data.append(metrics_iter)

        if len(products_plot) > 0:
            if verbose:
                print(metrics_iter)
            fig = starcoplot.plot_batch(plume_data, input_products=dataloader.dataset.input_products,
                                        products_plot=products_plot,
                                        figsize_ax=(4, 4))
            if show_plots:
                plt.show(fig)

            if path_save_results is not None:
                path_save_images = os.path.join(path_save_results, "images")
                if not os.path.exists(path_save_images):
                    os.makedirs(path_save_images)
                if not skip_saving_plots:
                    plt.savefig(os.path.join(path_save_images, f"{metrics_iter['id']}.png"), format="png")

            plt.close()

    out_data = pd.DataFrame(out_data).set_index("id")
    
    # Compute metrics by difficulty
    out_data["has_plume"] = out_data["label_pixels_plume"] > 0
    out_data["difficulty"] = out_data["label_pixels_plume"].apply(lambda x: "easy" if x > 1000 else "hard")
    
    metrics_by_difficulty = out_data.groupby(["has_plume","difficulty"])[["TP","FP","TN","FN"]].sum()
    metrics_by_difficulty["total"] = metrics_by_difficulty.sum(axis=1)
    metrics_by_difficulty["frac_total"] = metrics_by_difficulty["total"] / metrics_by_difficulty["total"].sum()
    
    # Dict to save results
    metrics = {}
    
    item = metrics_by_difficulty.loc[(False,"hard")]
    metrics["FPR_no_plume"] = item.FP / (item.FP + item.TN)
    metrics["frac_total_easy"] = item.frac_total
    
    for str_diff in ["easy", "hard"]:
        item = metrics_by_difficulty.loc[(True, str_diff)]
        cm_diff = torch.tensor([[item.TN, item.FP], [item.FN, item.TP]],requires_grad=False)
        for f in starcopmetrics.METRICS_CONFUSION_MATRIX:
            metrics[f"{f.__name__}_{str_diff}"] = f(cm_diff).item()
        metrics[f"frac_total_{str_diff}"] = item.frac_total

    # Compute aggregated metrics
    cm = confusion_metric.compute().cpu()
    for fun in starcopmetrics.METRICS_CONFUSION_MATRIX:
        metrics[fun.__name__] = fun(cm).item()

    metrics["confusion_matrix"] = cm

    # Compute classification metrics
    cm_classification_metric = torchmetrics.ConfusionMatrix(num_classes=2)
    cm_classification_metric(torch.from_numpy(out_data["pred_classification"].values).long(),
                             torch.from_numpy(out_data["has_plume"].values).long())
    cm_classification = cm_classification_metric.compute()

    for fun in starcopmetrics.METRICS_CONFUSION_MATRIX:
        metrics[f"classification_{fun.__name__}"] = fun(cm_classification).item()

    metrics["classification_confusion_matrix"] = cm_classification

    # PR curve
    metrics["thresholded"] = []
    for cm_thres in confusion_metric_thresholds:  # thresholds from high to low
        cm_iter_thres = cm_thres["confusion_matrix"].compute().cpu()
        dict_thres = {"threshold": cm_thres["threshold"], "confusion_matrix": cm_iter_thres}
        for fun in [starcopmetrics.precision, starcopmetrics.recall, starcopmetrics.TPR, starcopmetrics.FPR]:
            dict_thres[fun.__name__] = fun(cm_iter_thres)

        metrics["thresholded"].append(dict_thres)

    # Save stuff
    if path_save_results is not None:
        out_data.to_csv(os.path.join(path_save_results, f"results.csv"))
        with open(os.path.join(path_save_results, f"results_agg.json"), "w") as fh:
            json.dump(metrics, fh, cls=CustomJSONEncoder)

        if path_save_results_remote is not None:
            import fsspec
            fs = fsspec.filesystem("gs")

            if not path_save_results_remote.endswith("/"):
                path_save_results_remote = path_save_results_remote + "/"
            if path_save_results.endswith("/"):
                path_save_results = path_save_results_remote[:-1]

            fs.put(path_save_results, path_save_results_remote, recursive=True)

    return out_data, metrics

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, obj_to_encode):
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
        # if isinstance(obj_to_encode, Polygon):
        #     return mapping(obj_to_encode)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat()
        # if isinstance(obj_to_encode, datetime):
        #     return obj_to_encode.isoformat()
        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)

