from torch import Tensor
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from typing import Dict, Optional, Union, List
import wandb
import starcop.plot as starcoplot
from starcop.torch_utils import to_device


def numpy_tensor(x: Tensor, transpose : bool = True):
    if not torch.is_tensor(x):
        raise ValueError(f"Expected tensor found {x}")

    x = to_device(x, device=torch.device('cpu')).detach().numpy()
    if transpose:
        if x.ndim == 3 and (x.shape[0] > 1):
            x = x.transpose(1, 2, 0)
        if x.ndim == 4 and (x.shape[1] > 1):
            x = x.transpose(0, 2, 3, 1)

    return x


class ImageLogger(Callback):
    '''
    Based on
    https://colab.research.google.com/drive/12oNQ8XGeJFMiSBGsQ8uth8TaghVBC9H3
    '''

    def __init__(self, batch_train : Dict[str,Tensor], batch_test : Dict[str,Tensor], input_products:List[str], products_plot:List[str]) -> None:
        super().__init__()
        self.batch_train = batch_train
        self.batch_test = batch_test
        self.input_products = input_products
        self.products_plot = products_plot

    def on_train_epoch_end(self, trainer : Trainer, model : LightningModule,  unused: Optional = None) -> None:
        if trainer.logger:
            log = trainer.logger.experiment.log
            log(self.on_split_epoch_end(self.batch_train, model, 'train'), commit=False)

    def on_validation_epoch_end(self, trainer : Trainer, model : LightningModule) -> None:
        if trainer.logger:
            log = trainer.logger.experiment.log
            log(self.on_split_epoch_end(self.batch_test, model, 'val'), commit=False)
    #
    # def on_test_epoch_end(self, trainer : Trainer, f : LightningModule) -> None:
    #     if trainer.logger:
    #         log = trainer.logger.experiment.log
    #         log(self.on_split_epoch_end(self.batch_test, trainer, f, 'test'), commit=False)

    def on_split_epoch_end(self, batch, model : LightningModule, data_split_name : str) -> Dict:
        with torch.no_grad():
            batch_device = to_device(batch, model.device)
            batch_device_with_preds = model.batch_with_preds(batch_device)

        fig, ax = starcoplot.plot_batch(to_device(batch_device_with_preds,torch.device("cpu")),
                                    self.input_products, self.products_plot)

        return {f"{data_split_name}_batch": fig}

