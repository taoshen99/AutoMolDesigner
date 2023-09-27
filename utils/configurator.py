import os
import json
from torch import device
from torch.cuda import is_available

class Configurator:
    def __init__(self, config_path=None):
        self.config_path = config_path
    
    def configurate(self):
        if self.config_path is not None:
            with open(self.config_path, 'r') as f:
                config_dct = json.load(f)
            self.__dict__.update(config_dct)

        self.prj_dir = os.path.join('projects', self.prj_name)

        if os.path.exists(self.prj_dir):
            raise Exception(f'Project {self.prj_name} exists!')
        elif 'sampling_num' in self.__dict__.keys():
            os.makedirs(self.prj_dir)
        elif 'model_dir' in self.__dict__.keys():
            os.makedirs(self.prj_dir)
        elif 'quality' in self.__dict__.keys():
            self.model_dir = os.path.join(self.prj_dir, 'ag_models')
            os.makedirs(self.prj_dir)  
        else:
            self.log_dir = os.path.join(self.prj_dir, 'logs')
            os.makedirs(self.prj_dir)
            os.makedirs(self.log_dir)
    
    @staticmethod
    def check_device():
        if is_available():
            DEVICE = device("cuda")
            print("Model running on CUDA-enabled GPU!")
        else:
            DEVICE = device("cpu")
            print("Model running on CPU! If you have CUDA-enabled GPU, please check whether PyTorch-GPU has been correctly configured!")
        
        return DEVICE