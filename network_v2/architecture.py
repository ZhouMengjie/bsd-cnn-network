import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from torchvision import models
import sys
from Resnet50_v1 import cnn_resnet50
from Resnet18_v1 import resnet18

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class EmbeddingNet(nn.Module):
    def __init__(self, cfg, l2_normalize = True, scale=3):
        super(EmbeddingNet, self).__init__()       # Input shape of 128
        # Rest of model (Projection)

        self.relu = nn.ReLU(inplace=False)
        self.sm = nn.Softsign()

        #self.drop1 = nn.Dropout2d(0.5)
        #self.drop2 = nn.Dropout2d(0.5)
        #self.drop3 = nn.Dropout2d(0.5)
        #self.drop4 = nn.Dropout2d(0.5)

        self.xfc1 = nn.Linear(cfg.clusters*512, 512)
        self.xfc2 = nn.Linear(512,cfg.embedding_dim)
        self.xbn1 = nn.BatchNorm1d(num_features=cfg.clusters*512)
        self.xbn2 = nn.BatchNorm1d(num_features=512)
        self.xbn3 = nn.BatchNorm1d(num_features=cfg.embedding_dim)

        self.yfc1 = nn.Linear(cfg.clusters*512*4, 1024)
        self.yfc2 = nn.Linear(1024,cfg.embedding_dim)
        self.ybn1 = nn.BatchNorm1d(num_features=cfg.clusters*512*4)
        self.ybn2 = nn.BatchNorm1d(num_features=1024)
        self.ybn3 = nn.BatchNorm1d(num_features=cfg.embedding_dim)
        self.cfg = cfg

    def forward(self,x,y):
        # Branch 1 for x
        x = self.xbn1(x)
        x = self.relu(x)        
        #x = self.drop1(x) 

        x = self.xfc1(x)
        x = self.xbn2(x)
        x = self.relu(x)

        #x = self.drop2(x)
        x = self.xfc2(x)
        #x = self.sm(x)        

        # Branch 2 for y
        y = self.ybn1(y)
        y = self.relu(y)
        #y = self.drop3(y) 
        y = self.yfc1(y)
        y = self.ybn2(y)        
        y = self.relu(y)
        #y = self.drop4(y) 
        y = self.yfc2(y)
        #y = self.sm(y)

        if self.cfg.l2_normalization:
            x_norm = x.norm(p=2, dim=1, keepdim=True)
            x_l2_normalized = x.div(x_norm)

            y_norm = y.norm(p=2, dim=1, keepdim=True)
            y_l2_normalized = y.div(y_norm)

            return self.cfg.scale * x_l2_normalized, self.cfg.scale * y_l2_normalized
        else:
            return x, y

class MyModel(nn.Module):
    def __init__(self, cfg): 
        super(MyModel, self).__init__()       # Input shape of 128
        def get_activation(name):
            def hook(model, input_img, output):
                self.activations[name] = output
            return hook

        def weights_init(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.cfg = cfg

        # Load gsv_model
        model_file = 'resnet50_places365.pth.tar'
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

        self.gsv_model = cnn_resnet50()
        self.gsv_model.load_state_dict(state_dict, strict=False)

        for param in self.gsv_model.parameters():
            param.requires_grad = False
        for param in self.gsv_model.layer5.parameters():
            param.requires_grad = True
            
        
        # Load tile model
        self.tile_model = resnet18(pretrained=True)
        
        # Vlad net
        self.x_vlad = NetVLAD(num_clusters=cfg.clusters,dim=512)
        self.y_vlad = NetVLAD(num_clusters=cfg.clusters,dim=512)

        # Embedding model -----
        self.embedding_model = EmbeddingNet(cfg)
        self.embedding_model.apply(weights_init)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
        #     self.gsv_model = nn.DataParallel(self.gsv_model)
        #     self.tile_model = nn.DataParallel(self.tile_model)
        #     self.embedding_model = nn.DataParallel(self.embedding_model)

        
    def forward(self, X, Yf, Yl, Yr, Yb):
        X = self.tile_model(X)
        X = self.x_vlad(X)        

        # Fordward pass for gsvs
        Yf = self.gsv_model(Yf)
        Yf = self.y_vlad(Yf)

        Yl = self.gsv_model(Yl)
        Yl = self.y_vlad(Yl)        

        Yr = self.gsv_model(Yr)
        Yr = self.y_vlad(Yr)        

        Yb = self.gsv_model(Yb)
        Yb = self.y_vlad(Yb)        

        Y = torch.cat((Yf,Yl, Yr, Yb), dim=1)
        
        X_out, Y_out = self.embedding_model.forward(X, Y)
        return X_out, Y_out
