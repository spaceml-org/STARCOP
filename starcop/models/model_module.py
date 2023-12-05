import torch
import torch.nn
import wandb
import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Tuple
from .utils import losses, metrics
from starcop.utils import get_filesystem
from .architectures.unet import UNet, UNet_dropout
from .architectures.baselines import SingleConv, SimpleCNN
import torchmetrics
from starcop.data.normalizer_module import DataNormalizer
from starcop import metrics


def to_tensor(v):
    if isinstance(v,list):
        v=torch.tensor(v)
    elif isinstance(v,np.ndarray):
        v=torch.from_numpy(v)
    return v


class ModelModule(pl.LightningModule):

    def __init__(self, settings):
        super().__init__()
        self.save_hyperparameters()
        self.settings_model = settings.model
        self.settings_wandb = settings.wandb
        self.normalizer = DataNormalizer(settings)

        self.num_classes = self.settings_model.num_classes
        self.num_channels = len(settings.dataset.input_products)
        
        architecture = self.settings_model.model_type
        self.network = configure_architecture(architecture, self.num_channels, self.num_classes, self.settings_model)

        # learning rate params
        self.lr = self.settings_model.lr
        self.lr_decay = self.settings_model.lr_decay
        self.lr_patience = self.settings_model.lr_patience
        
        self.loss_name = self.settings_model.loss

        use_weight_loss = "use_weight_loss" not in settings.dataset or settings.dataset.use_weight_loss
        if self.settings_model.loss == 'l1':
            self.loss_function = losses.l1
            self.loss_name = "l1_loss"
        elif self.settings_model.loss == 'mse':
            self.loss_function = losses.mse
            self.loss_name = "mse_loss"
        elif self.settings_model.loss == "BCEWithLogitsLoss":
            self.reduction = "none" if use_weight_loss else "mean"
            self.pos_weight = torch.nn.Parameter(torch.tensor(float(self.settings_model.pos_weight)),
                                                 requires_grad=False)
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight,
                                                            reduction=self.reduction)

        # Configure metrics based on settings_model.model_mode: "segmentation_output" # regression_output
        if self.settings_model.model_mode == "segmentation_output":
            self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, task="binary")
            self.classification_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, task="binary")


        elif self.settings_model.model_mode == "regression_output":
            raise NotImplementedError("Not implemented yet")

    def training_step(self, batch: Dict, batch_idx) -> float:
        x, y = batch["input"], batch["output"]

        if self.reduction == "none":
            weight_loss = batch["weight_loss"]

        predictions = self.forward(x) # (B, 1, H, W)
        loss = self.loss_function(predictions, self.normalizer.normalize_y(y)) # (B, 1, W, H)

        if self.reduction == "none":
            loss = torch.mean(loss * weight_loss)

        if (batch_idx % 100) == 0:
            self.log(f"train_{self.loss_name}", loss)
        
        # if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
        #     with torch.no_grad():
        #         self.log_images(x, y, predictions, prefix="train_")
            
        return loss
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_channels, H, W) input tensor

        Returns:
            (B, 1, H, W) prediction of the network
        """
        return self.network(self.normalizer.normalize_x(x))

    def pred_classification(self, pred_binary:torch.Tensor) -> torch.Tensor:
        return pred_classification(pred_binary)
    
    def log(self, *args, **kwargs):
        try:
            super().log(*args,**kwargs)
        except Exception as e:
            print(f"Bug logging {e}")
        

    def val_step(self, batch, batch_idx:int, prefix:str="val"):
        x, y = batch["input"], batch["output"]

        predictions = self.forward(x)
        y = self.normalizer.normalize_y(y)
        loss = self.loss_function(predictions, y)
        
        if self.reduction == "none":
            weight_loss = batch["weight_loss"]
            loss = torch.mean(loss * weight_loss)

        self.log(f"{prefix}_loss", loss, on_epoch=True)
        
        if self.settings_model.model_mode == "segmentation_output":
            pred_binary = (predictions >= 0).long() # (B, 1, H, C)

            y_long = y.long() # (B, 1, H, C)

            self.confusion_matrix.update(pred_binary, y_long)

            y_classification = batch["has_plume"]
            y_classification = y_classification[:, None]

            pred_classification = self.pred_classification(pred_binary)

            self.classification_confusion_matrix.update(pred_classification, y_classification)

        # Visualizations
        # if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
        #     self.log_images(x, y, predictions,prefix=prefix)

    def validation_step(self, batch, batch_idx: int):
        return self.val_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx: int) :
        return self.val_step(batch, batch_idx, prefix="test")

    def val_epoch_end(self, outputs, prefix):
        outs = {}
        cm = self.confusion_matrix.compute()

        for fun in metrics.METRICS_CONFUSION_MATRIX:
            self.log(f'{prefix}_{fun.__name__}', fun(cm))

        self.confusion_matrix.reset()

        if self.settings_model.model_mode == "segmentation_output":
            cm = self.classification_confusion_matrix.compute()

            for fun in metrics.METRICS_CONFUSION_MATRIX:
                self.log(f'{prefix}_classification_{fun.__name__}', fun(cm))

            self.classification_confusion_matrix.reset()

        return outs
    
    def validation_epoch_end(self, outputs) -> None:
        self.val_epoch_end(outputs, prefix="val")

    def test_epoch_end(self, outputs) -> None:
        self.val_epoch_end(outputs, prefix="test")

    def configure_optimizers(self):
        if self.settings_model.optimizer == "adam":
            optimizer = torch.optim.Adam(self.network.parameters(), self.lr)
        else:
            raise Exception(f'No optimizer implemented for : {self.settings_model.optimizer}')

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=self.lr_decay, verbose=True,
                                                               patience=self.lr_patience)

        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    def debug(self):
        print("Model debug:")
        print(self)

    def batch_with_preds(self, batch):
        logits = self(batch["input"])
        pred = torch.sigmoid(logits)

        batch = batch.copy()
        batch["input_norm"] = self.normalizer.normalize_x(batch["input"])
        batch["output_norm"] = self.normalizer.normalize_y(batch["output"])

        batch["prediction"] = pred
        batch["logits"] = logits
        if self.reduction == "none":
            batch["loss_per_pixel"] = self.loss_function(logits, batch["output_norm"])
            batch["loss_per_pixel_weighted"] = batch["weight_loss"] * batch["loss_per_pixel"]
        batch["pred_binary"] = (pred > .5).long()
        batch["differences"] = differences(batch["pred_binary"], batch["output_norm"].long())
        batch["pred_classification"] = self.pred_classification(batch["pred_binary"])

        return batch

def pred_classification(pred_binary:torch.Tensor) -> torch.Tensor:
    n_pixels = (10 * np.prod(tuple(pred_binary.shape[-2:]))) / (64 ** 2)
    return (torch.sum(pred_binary, dim=(-1, -2)) > n_pixels).long()  # (B, 1)


METRIC_MODE = {
    # In min mode, lr will be reduced when the quantity monitored has stopped decreasing
    "val_l1_loss": "min",
    "train_l1_loss": "min",
    "val_mse_loss": "min",
    "train_mse_loss": "min",
}


def configure_architecture(architecture, num_channels, num_classes, extra_settings_model):

    # if architecture == 'unet':
    #     model = UNet(num_channels, num_classes)
    #
    # elif architecture == 'cnn':
    #     model = SimpleCNN(num_channels, num_classes)
    #
    # elif architecture == 'single':
    #     model = SingleConv(num_channels, num_classes)
    #
    # elif architecture == 'unet_dropout':
    #     model = UNet_dropout(num_channels, num_classes)

    if architecture == 'unet_semseg':
        
        import segmentation_models_pytorch as smp
        BACKBONE = extra_settings_model.semseg_backbone
        PRETRAINED = 'imagenet' if num_channels == 3 else None

        model = smp.Unet(
            encoder_name=BACKBONE,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=PRETRAINED,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=num_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
            activation=None,      # activation function, default is None
            # encoder_depth=4 # Depth parameter specify a number of downsampling operations in encoder, so you can make your model lighter if specify smaller depth
        )
        
    else:
        raise Exception(f'No model implemented for model_type: {architecture}')

    return model

def load_weights(path_weights:str, map_location="cpu"):
    fs = get_filesystem(path_weights)
    if fs.exists(path_weights):
        with fs.open(path_weights, "rb") as fh:
            weights = torch.load(fh, map_location=map_location)

        return weights

    raise ValueError(f"Pretrained weights file: {path_weights} does not exists")

def differences(y_pred_binary: torch.Tensor, y_gt:torch.Tensor) -> torch.Tensor:
    return 2 * y_pred_binary.long() + (y_gt == 1).long()