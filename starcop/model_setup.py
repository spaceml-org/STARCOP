import os
from .models.model_module import ModelModule, load_weights
from .models.model_module_regression import ModelModuleRegression

def get_model(settings, experiment_name):
    
    if settings.model.model_mode == "segmentation_output":
        model = ModelModule(settings)
        
    elif settings.model.model_mode == "regression_output":
        model = ModelModuleRegression(settings)

    if settings.model.test:
        assert experiment_name is not None, f"Expermient name must be set on test or deploy mode"

        path_to_models = os.path.join(settings.model.model_folder, experiment_name, "model.pt").replace("\\", "/")
        model.load_state_dict(load_weights(path_to_models))
        print(f"Loaded model weights: {path_to_models}")

    return model
