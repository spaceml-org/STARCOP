import torch
import torch.nn
import wandb
import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Tuple
from .utils import losses, metrics
from starcop.utils import get_filesystem
from .architectures.unet import UNet, UNet_dropout
from .architectures.baselines import SingleConv, SimpleCNN, SimpleCNN_v2, SimpleCNN_v3
import torchmetrics
from starcop.data.normalizer_module import DataNormalizer
from starcop import metrics


class ModelModuleRegression(pl.LightningModule):

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

        if self.settings_model.loss == 'l1':
            self.loss_function = losses.l1
            self.loss_name = "l1_loss"
        elif self.settings_model.loss == 'mse':
            self.loss_function = losses.mse
            self.loss_name = "mse_loss"
        
        if self.settings_model.model_mode == "regression_output":
            print("in regression mode")
        else:
            print("this model module should be only used with regression!")
            assert False
            
            
        # Additional settings ....
        self.inhibit_normalisation = True
        
        print("inhibiting normalisation:" , self.inhibit_normalisation)

    def training_step(self, batch: Dict, batch_idx) -> float:
        x, y = batch["input"], batch["output"]

        predictions = self.forward(x) # (B, 1, H, W)
        if not self.inhibit_normalisation:
            loss = self.loss_function(predictions, self.normalizer.normalize_y(y)) # (B, 1, W, H)
        else:
            loss = self.loss_function(predictions, y) # (B, 1, W, H)

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
        if not self.inhibit_normalisation:
            return self.network(self.normalizer.normalize_x(x))
        else:
            return self.network(x)
    
    def log(self, *args, **kwargs):
        try:
            super().log(*args,**kwargs)
        except Exception as e:
            print(f"Bug logging {e}")
        

    def val_step(self, batch, batch_idx:int, prefix:str="val"):
        x, y = batch["input"], batch["output"]

        predictions = self.forward(x)
        if not self.inhibit_normalisation:
            y = self.normalizer.normalize_y(y)
        loss = self.loss_function(predictions, y)
        
        self.log(f"{prefix}_loss", loss, on_epoch=True)
        
        # Visualizations
        # if (batch_idx == 0) and (self.logger is not None) and isinstance(self.logger, WandbLogger):
        #     self.log_images(x, y, predictions,prefix=prefix)

    def validation_step(self, batch, batch_idx: int):
        return self.val_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx: int) :
        return self.val_step(batch, batch_idx, prefix="test")

    def val_epoch_end(self, outputs, prefix):
        outs = {}
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
        # might not make too much sense for now ...
        pred = self(batch["input"])

        batch = batch.copy()
        
        batch["input_norm"] = self.normalizer.normalize_x(batch["input"])
        batch["output_norm"] = self.normalizer.normalize_y(batch["output"])

        batch["prediction"] = pred
        batch["logits"] = pred # no logits here ...

        if not self.inhibit_normalisation:
            batch["differences"] = differences(batch["prediction"], batch["output_norm"].float())
        else:
            batch["differences"] = differences(batch["prediction"], batch["output"].float())
            
        return batch



def configure_architecture(architecture, num_channels, num_classes, extra_settings_model):

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
    elif architecture == 'cnn_v1':
        model = SimpleCNN(num_channels, num_classes)
    elif architecture == 'cnn_v2':
        model = SimpleCNN_v2(num_channels, num_classes)
    elif architecture == 'cnn_v3':
        model = SimpleCNN_v3(num_channels, num_classes)
        
        
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

def differences(y_pred: torch.Tensor, y_gt:torch.Tensor) -> torch.Tensor:
    return y_pred - y_gt