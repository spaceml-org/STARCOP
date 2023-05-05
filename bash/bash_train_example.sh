# Examples of bash commands used to train the configurations mentioned in the paper:
# ================================
# HyperSTARCOP mag1c + rgb

python -m scripts.train dataset.input_products=["mag1c","TOA_AVIRIS_640nm","TOA_AVIRIS_550nm","TOA_AVIRIS_460nm"] model.model_type='unet_semseg' model.pos_weight=1 experiment_name="HyperSTARCOP_magic_rgb" dataloader.num_workers=4 dataset.use_weight_loss=True dataset.train_csv="train.csv" training.val_check_interval=0.5 training.max_epochs=15 products_plot=["rgb_aviris","mag1c","label","pred","differences"] dataset.weight_sampling=True

# ================================
# HyperSTARCOP mag1c only

python -m scripts.train dataset.input_products=["mag1c"] model.model_type='unet_semseg' model.pos_weight=1 experiment_name="HyperSTARCOP_magic_only" dataloader.num_workers=4 dataset.use_weight_loss=True dataset.train_csv="train.csv" training.val_check_interval=0.5 training.max_epochs=15 products_plot=["mag1c","label","pred","differences"] dataset.weight_sampling=True

# ================================
# MultiSTARCOP Varon ratios

python -m scripts.train dataset.input_products=["ratio_wv3_B7_B5_varon21_sum_c_out","ratio_wv3_B8_B5_varon21_sum_c_out","ratio_wv3_B7_B6_varon21_sum_c_out"] model.model_type='unet_semseg' experiment_name="MultiSTARCOP_Varon" training.max_epochs=15 dataset.train_csv='train.csv' products_plot=["wv3_ratios_varon_b7b5","wv3_ratios_varon_b8b5","wv3_ratios_varon_b7b6","label","pred","differences"] model.pos_weight=15 dataset.use_weight_loss=True dataset.weight_sampling=True

# ================================
# MultiSTARCOP Sanchez ratios

python -m scripts.train dataset.input_products=["ratio_wv3_B7_B7MLR_SanchezGarcia22_sum_c_out","ratio_wv3_B8_B8MLR_SanchezGarcia22_sum_c_out","TOA_WV3_SWIR1"] model.model_type='unet_semseg' experiment_name="MultiSTARCOP_Sanchez" training.max_epochs=15 dataset.train_csv='train.csv' products_plot=["wv3_ratios_sanchez_b7b7mlr","wv3_ratios_sanchez_b8b8mlr","wv3_b1","label","pred","differences"] model.pos_weight=15 dataset.use_weight_loss=True dataset.weight_sampling=True

# ================================
# MultiSTARCOP Varon+Sanchez ratios

python -m scripts.train dataset.input_products=["ratio_wv3_B7_B5_varon21_sum_c_out","ratio_wv3_B8_B5_varon21_sum_c_out","ratio_wv3_B7_B7MLR_SanchezGarcia22_sum_c_out"] model.model_type='unet_semseg' experiment_name="MultiSTARCOP_Varon_Sanchez" training.max_epochs=15 dataset.train_csv='train.csv' products_plot=["wv3_ratios_varon_b7b5","wv3_ratios_varon_b8b5","wv3_ratios_sanchez_b7b7mlr","label","pred","differences"] model.pos_weight=15 dataset.use_weight_loss=True dataset.weight_sampling=True


# ================================