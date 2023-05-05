import matplotlib
matplotlib.use('agg')

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from starcop.dataset_setup import get_dataset
from starcop.model_setup import get_model
from starcop.data.data_logger import ImageLogger
from hydra.utils import get_original_cwd
import logging
import fsspec
from starcop.validation import  run_validation
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(settings : DictConfig) -> None:
    experiment_path = os.getcwd()
    folder_relative_name = experiment_path.replace(get_original_cwd(), "") #remove beginning of path
    
    log = logging.getLogger(__name__)

    checkpoint_path = os.path.join(experiment_path, "checkpoint").replace("\\", "/")
    if not experiment_path.startswith("gs://"):
        os.makedirs(experiment_path, exist_ok=True)

    if not checkpoint_path.startswith("gs://"):
        os.makedirs(checkpoint_path, exist_ok=True)

    # Set up remote path
    if folder_relative_name.startswith("/"):
        folder_relative_name = folder_relative_name[1:]
    if not folder_relative_name.endswith("/"):
        folder_relative_name += "/"
    remote_path = os.path.join("gs://starcop/", folder_relative_name)

    OmegaConf.set_struct(settings, False)
    settings["experiment_path"] = experiment_path
    settings["experiment_path"] = remote_path

    log.info(f"trained models will be save at {experiment_path}")
    log.info(f"At the end of training, models will be copied to {remote_path}")
    
    plt.ioff()

    # LOGGING SETUP
    log.info("SETTING UP LOGGERS")
    wandb_logger = WandbLogger(
        name=settings.experiment_name,
        project=settings.wandb.wandb_project, 
        entity=settings.wandb.wandb_entity,
    )
    # wandb.config.update(settings)
    wandb_logger.experiment.config.update(settings)

    settings["wandb_logger_version"] = wandb_logger.version
    OmegaConf.set_struct(settings, True)

    log.info(f"Settings dump:{OmegaConf.to_yaml(settings)}")
    log.info(f"Using matplotlib backend: {matplotlib.get_backend()}")

    # ======================================================
    # EXPERIMENT SETUP
    # ======================================================
    # Seed
    seed_everything((None if settings.seed == "None" else settings.seed))
    
    # DATASET SETUP
    log.info("SETTING UP DATASET")
    data_module = get_dataset(settings)
    data_module.prepare_data()

    # MODEL SETUP
    log.info("SETTING UP MODEL")
    settings.model.test = False
    settings.model.train = True
    model = get_model(settings, settings.experiment_name)
    
    # CHECKPOINTING SETUP
    log.info("SETTING UP CHECKPOINTING")
    
    metric_monitor ="val_loss" 
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=True,
        verbose=True,
        monitor=metric_monitor,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor=metric_monitor,
        patience=settings.model.early_stopping_patience,
        strict=False,
        verbose=False,
        mode="min"
    )

    # Images for logs from the first batch
    batch_train = next(iter(data_module.train_plot_dataloader(batch_size=settings.plot_samples)))
    batch_test = next(iter(data_module.test_plot_dataloader(batch_size=settings.plot_samples)))

    il = ImageLogger(batch_train=batch_train,
                     batch_test=batch_test, products_plot=settings.products_plot,
                     input_products=settings.dataset.input_products)

    callbacks = [checkpoint_callback, il]
    
    # TRAINING SETUP
    log.info("START TRAINING")
    
    # See: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=callbacks,
        auto_select_gpus=True,
        default_root_dir=experiment_path,
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        auto_lr_find=False,
        benchmark=False,
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        max_epochs=settings.training.max_epochs,
        # check_val_every_n_epoch=settings.training.val_every,
        val_check_interval=settings.training.val_check_interval,
        # Pass a float in the range [0.0, 1.0] to check after a fraction of the training epoch. Pass an int to check after a fixed number of training batches. An int value can only be higher than the number of training batches when
        log_every_n_steps=settings.training.train_log_every_n_steps,
        resume_from_checkpoint=checkpoint_path if settings.resume_from_checkpoint else None
    )
    
    trainer.fit(model, data_module)

    # Save model
    trainer.save_checkpoint(os.path.join(experiment_path, "final_checkpoint_model.ckpt"))

    # Copy stuff to remote path
    try:
        fs = fsspec.filesystem("gs")
        fs.put(experiment_path, remote_path, recursive=True)
    except:
        print("Failed when trying to copy the model to the remote path", remote_path)

    log.info("Running validation of val data")
    dataloader_val = data_module.test_plot_dataloader(batch_size=1,num_workers=data_module.num_workers)
    run_validation(model, dataloader_val, products_plot=settings.products_plot,
                   show_plots=False, verbose=False,
                   path_save_results=os.path.join(remote_path,"validation"))

    log.info("Running validation of train data")
    dataloader_train =DataLoader(data_module.train_dataset_non_tiled, batch_size=1,
                                 num_workers=data_module.num_workers, shuffle=False)

    run_validation(model, dataloader_train, products_plot=settings.products_plot,
                   show_plots=False, verbose=False,
                   path_save_results=os.path.join(remote_path, "train"))
    log.info(f"Finish: results copied to {remote_path}")



if __name__ == "__main__":
    train()
