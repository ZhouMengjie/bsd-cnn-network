import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from torchvision import models
import sys
from Resnet import resnet18,resnet50

class MyModel(nn.Module):
    def __init__(self, cfg): 
        super(MyModel, self).__init__()       # Input shape of 128
        # def get_activation(name):
        #     def hook(model, input_img, output):
        #         self.activations[name] = output
        #     return hook

        def weights_init(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.cfg = cfg

        # Load gsv_model
        # self.gsv_model = models.__dict__['resnet18'](num_classes=16)
        self.gsv_model = resnet18()
        self.fc = nn.Linear(512 * 4, 16)

        if cfg.mode == 'train':
            model_file = 'resnet18_places365.pth.tar'
            if not os.access(model_file, os.W_OK):
                weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                os.system('wget ' + weight_url)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k,'fc.bias' ,'fc1.bias'): v for k,v in state_dict.items()}
            state_dict = {str.replace(k,'fc.weight' ,'fc1.weight'): v for k,v in state_dict.items()}
            self.gsv_model.load_state_dict(state_dict, strict=False)
            self.fc.apply(weights_init)
            # self.gsv_model.fc.apply(weights_init)
       
    def forward(self, Yf, Yl, Yr, Yb):
        # Fordward pass for gsvs
        Yf = self.gsv_model(Yf)
        Yl = self.gsv_model(Yl)    
        Yr = self.gsv_model(Yr)    
        Yb = self.gsv_model(Yb)     
        Y = torch.cat((Yf,Yl, Yr, Yb), dim=1)
        out = self.fc(Y)  

        return out
