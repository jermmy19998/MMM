import os
import pdb
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



        
class SDL(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.25):
        """
        Stable Dense Layer (SDL)
        
        This layer consists of a fully connected linear transformation 
        followed by SELU activation and Alpha Dropout, which helps 
        stabilize training by preserving self-normalizing properties.

        Args:
            in_dim (int): Number of input features.
            out_dim (int): Number of output features.
            dropout_rate (float, optional): Dropout rate for AlphaDropout. Default is 0.25.
        """
        super(SDL, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate)
        )

    def forward(self, x):
        """
        Forward pass through the SDL layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        return self.layers(x)





class DAM(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100):
        """
        Dual Affine Mapper (DAM)
        
        This module performs dual affine transformations on two input modalities 
        (e.g., image and omic data) and combines them through element-wise operations.
        It aims to learn modality-specific affine mappings and interaction patterns.

        Args:
            in_dim (int, optional): Dimensionality of input features. Default is 1024.
            out_dim (int, optional): (Unused) Placeholder for potential output dimension. Default is 100.
        """
        super(DAM, self).__init__()
        self.in_dim = in_dim
        self.i_layer = nn.Linear(in_dim, 2 * in_dim)  # Affine transformation for x_i
        self.o_layer = nn.Linear(in_dim, 2 * in_dim)  # Affine transformation for x_o

    def forward(self, x_i, x_o):
        """
        Forward pass through the DAM module.

        Args:
            x_i (torch.Tensor): Input tensor from the first modality (e.g., image) of shape (batch_size, in_dim).
            x_o (torch.Tensor): Input tensor from the second modality (e.g., omic) of shape (batch_size, in_dim).

        Returns:
            tuple of torch.Tensor:
                - mix_image (torch.Tensor): Transformed output for the first modality of shape (batch_size, in_dim).
                - mix_omic (torch.Tensor): Transformed output for the second modality of shape (batch_size, in_dim).
        """
        head_image, tail_image = self.i_layer(x_i).chunk(2, dim=-1)  
        head_omic, tail_omic = self.o_layer(x_o).chunk(2, dim=-1)  

        # Cross-modal interaction through element-wise operations
        mix_image = (head_omic * tail_image) + (tail_image + head_omic)
        mix_omic = (head_image * tail_omic) + (tail_omic + head_image)
        return mix_image, mix_omic



        
        
class LinearRes(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.25):
        """
        Linear Residual Block (LinearRes)

        This module implements a simple residual connection over a linear transformation 
        with GELU activation. It is useful for stabilizing learning and improving gradient flow.

        Args:
            in_dim (int): Dimensionality of input features.
            out_dim (int): Dimensionality of output features.
            dropout_rate (float, optional): (Unused) Placeholder for potential dropout implementation. Default is 0.25.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        """
        Forward pass through the LinearRes module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim), with residual connection.
        """
        identity = x
        return self.layers(x) + identity


    
class GlobalChannelPooling(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        """
        Global Channel Pooling (GCP)

        This module performs global average and max pooling along the channel dimension,
        then combines the pooled features using a lightweight projection layer.
        It helps to aggregate important spatial information while reducing dimensionality.

        Args:
            in_dim (int, optional): Placeholder argument for future extensions. Default is None.
        """
        super().__init__()
        self.avg_pool = self.channel_avg_pool
        self.max_pool = self.channel_max_pool

        self.proj = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),  # Reduce depth to 1 channel
            nn.Sigmoid(),  # Normalize the output
        )

    def channel_avg_pool(self, x):
        """
        Compute channel-wise average pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Averaged feature map of shape (batch_size, 1, height, width).
        """
        return x.mean(dim=1, keepdim=True)

    def channel_max_pool(self, x):
        """
        Compute channel-wise max pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Max-pooled feature map of shape (batch_size, 1, height, width).
        """
        return x.max(dim=1, keepdim=True)[0]

    def forward(self, x):
        """
        Forward pass of the GlobalChannelPooling module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Projected feature map of shape (batch_size, 1, height, width).
        """
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        pooled_features = torch.cat((avg_pooled, max_pooled), dim=-1)  # Concatenate along channel dim
        projected = self.proj(pooled_features)  # Apply projection
        return projected

class Phi(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, dropout_rate=0.25):
        """
        Phi φ: integrates two modalities (e.g., image and omic data) by applying 
        global pooling, residual linear transformations, and an optional sequence length 
        adjustment mechanism.

        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

    def adjust_tensor(self, x_o, target_seq_len):
        """
        Adjusts the input tensor to match the target sequence length by repeating 
        and shuffling elements.

        Args:
            x_o (torch.Tensor): Input tensor of shape (batch_size, orig_seq_len, features).
            target_seq_len (int): Desired sequence length.

        Returns:
            torch.Tensor: Adjusted tensor of shape (batch_size, target_seq_len, features).
        """
        batch_size, orig_seq_len, features = x_o.shape
        repeat_times = (target_seq_len + orig_seq_len - 1) // orig_seq_len  
        x_repeated = x_o.repeat(1, repeat_times, 1)  
        x_adjusted = x_repeated[:, :target_seq_len, :]
        shuffled_indices = torch.randperm(target_seq_len)
        x_shuffled = x_adjusted[:, shuffled_indices, :]
        return x_shuffled

    def forward(self, x, target_seq_len=None):
        """
        Forward pass of the Phi module.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, in_dim) or (batch_size, seq_len, in_dim).
            target_seq_len (int, optional): If provided, adjusts the sequence length of x.

        Returns:
            torch.Tensor: Transformed feature map.
        """
        x = self.adjust_tensor(x, target_seq_len)
        return self.layer(x)



   
class MiT(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, dropout_rate=0.3):
        """
        Multi-Information Transmission (MiT)

        This module integrates two modalities (e.g., image and omic data) by applying 
        global pooling, residual linear transformations, and an optional sequence length 
        adjustment mechanism. It consists of two stages:
        - Stage 1: Adjusts the sequence length of the auxiliary modality (x_o).
        - Stage 2: Directly processes both modalities.

        Args:
            in_dim (int, optional): Input feature dimension. Default is 1024.
            out_dim (int, optional): Output feature dimension. Default is 1024.
        """
        super(MiT, self).__init__()

        self.lr_i = LinearRes(in_dim, out_dim, dropout_rate)  # Residual transformation for x_i
        self.lr_o = LinearRes(in_dim, out_dim, dropout_rate)  # Residual transformation for x_o
        self.global_i = GlobalChannelPooling(in_dim, in_dim)  # Global feature pooling
        self.global_o = GlobalChannelPooling(in_dim, in_dim)  # Global feature pooling

        

    def forward(self, x_i, x_o):
        """
        Forward pass through the MiT module.

        Args:
            x_i (torch.Tensor): Input tensor from the first modality (e.g., image) of shape (batch_size, seq_len, in_dim).
            x_o (torch.Tensor): Input tensor from the second modality (e.g., omic) of shape (batch_size, seq_len, in_dim).

        Returns:
            tuple of torch.Tensor:
                - mix_i (torch.Tensor): Transformed output for the first modality of shape (batch_size, seq_len, out_dim).
                - mix_o (torch.Tensor): Transformed output for the second modality of shape (batch_size, seq_len, out_dim).
        """
        # Compute global-pooled interactions
        mix_i = self.lr_i(x_i * self.global_i(x_o).squeeze(0))
        mix_o = self.lr_o(x_o * self.global_o(x_i).squeeze(0))

        return mix_i, mix_o




        
class MMM(nn.Module):
    def __init__(self,
                 in_dim=1024, 
                 hidden_dim=[256, 128], 
                 omic_sizes=[100, 200, 300, 400, 500, 600], 
                 n_classes=4):
        """
        Multi-Modal Multual Mixer (MMM)

        This model integrates multi-modal data (pathology images and omic features) using a combination 
        of Stable Dense Layers (SDL), Multi-Information Transmission (MiT) modules, Dual Affine Mapper (DAM), 
        and Harmony Classifier to predict survival hazards.

        Args:
            in_dim (int, optional): Input feature dimension for image modality. Default is 1024.
            hidden_dim (list, optional): Hidden dimensions for feature processing. Default is [256, 128].
            omic_sizes (list, optional): List of feature dimensions for different omic inputs. Default is [100, 200, 300, 400, 500, 600].
            n_classes (int, optional): Number of output classes for survival prediction. Default is 4.
        """
        super(MMM, self).__init__()

        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.omic_sdl_1 = nn.ModuleList([
            SDL(in_dim=omic_in_dim, out_dim=hidden_dim[0], dropout_rate=0.2) for omic_in_dim in omic_sizes
        ])
        self.image_sdl_1 = SDL(in_dim=in_dim, out_dim=hidden_dim[0], dropout_rate=0.2)

        self.phi = Phi(in_dim=hidden_dim[0], out_dim=hidden_dim[0])
        self.mit_1 = MiT(in_dim=hidden_dim[0], out_dim=hidden_dim[0],dropout_rate=0.3)
        
        self.sdl_i_2 = SDL(in_dim=hidden_dim[0], out_dim=hidden_dim[1], dropout_rate=0.3)
        self.sdl_o_2 = SDL(in_dim=hidden_dim[0], out_dim=hidden_dim[1], dropout_rate=0.3)

        self.dam = DAM(in_dim=hidden_dim[1])
        
        self.sdl_i_3 = SDL(in_dim=hidden_dim[1], out_dim=hidden_dim[1], dropout_rate=0.3)
        self.sdl_o_3 = SDL(in_dim=hidden_dim[1], out_dim=hidden_dim[1], dropout_rate=0.3)

        self.mit_2 = MiT(in_dim=hidden_dim[1], out_dim=hidden_dim[1], dropout_rate=0.35)

        self.classifier = self.classifier = nn.Linear(hidden_dim[1], n_classes)

    def forward(self, **kwargs): 
        """
        Forward pass of the MMM model.

        Args:
            kwargs (dict): Dictionary containing:
                - 'x_path' (torch.Tensor): Pathology image feature tensor of shape (batch_size, seq_len, in_dim).
                - 'x_omic1', 'x_omic2', ..., 'x_omic6' (torch.Tensor): Omic feature tensors.

        Returns:
            tuple:
                - hazards (torch.Tensor): Predicted hazard probabilities of shape (batch_size, n_classes).
                - survival (torch.Tensor): Cumulative survival probabilities of shape (batch_size, n_classes).
                - top_class (torch.Tensor): Predicted class indices of shape (batch_size, 1).
                - {} (dict): Placeholder for additional outputs.
        """
        x_image = kwargs['x_path'].unsqueeze(0) 
        x_image = self.image_sdl_1(x_image)  

        x_omic = torch.stack([sdl(kwargs[f'x_omic{i+1}']) for i, sdl in enumerate(self.omic_sdl_1)], dim=0).unsqueeze(0)
        x_omic = self.phi(x_omic, x_image.shape[1])

        out_image_1, out_omic_1 = self.mit_1(x_image, x_omic)
        
        out_image_1 = self.sdl_i_2(out_image_1)
        out_omic_1 = self.sdl_o_2(out_omic_1)

        # Cross-modal feature interaction
        mix_image_1, mix_omic_1 = self.dam(out_image_1, out_omic_1)
        
        mix_image_1 = self.sdl_i_3(mix_image_1)
        mix_omic_1 = self.sdl_o_3(mix_omic_1)

        out_image_2, out_omic_2 = self.mit_2(mix_image_1, mix_omic_1)

        # Culculate similarity matrix
        similarity = F.normalize(out_image_2.permute(0, 2, 1) @ out_omic_2) 
        
        # Classification
        logits = self.classifier(similarity.mean(1))
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)

        return hazards, survival, torch.topk(logits, 1, dim=1)[1], {}