from .data.datamodule import Permian2019DataModule

def get_dataset(settings):

    data_module = Permian2019DataModule(settings)

    return data_module