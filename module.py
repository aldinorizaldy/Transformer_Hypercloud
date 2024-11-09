# module.py
# note: skip SG and  neighbor embedding for now

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from util import sample_and_knn_group

from matplotlib import pyplot as plt


class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        # print('x_q size = ', x_q.size())
        x_k = self.k_conv(x)                   # [B, da, N]
        # print('x_k size = ', x_k.size())
        x_v = self.v_conv(x)                   # [B, de, N]
        # print('x_v size = ', x_v.size())
        # print('v_conv = ', self.v_conv.weight.size())

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        # print('energy size = ', energy.size())
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        # print('x_s size = ', x_s.size())
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        # print('x_s size = ', x_s.size())
        
        # residual
        x = x + x_s

        return x


# class SG(nn.Module):


# class NeighborEmbedding(nn.Module):


class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        # print('attention size = ', attention.size())
        # print('attention sum size = ', attention.sum(dim=1, keepdims=True).size())
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x

class ASCN(nn.Module):
    """
    Attentional ShapeContextNet module.
    """

    def __init__(self, channels):
        super(ASCN, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x_v + x_s # ASCN differs here !!

        return x

    
class PT(nn.Module):
    """
    PointTransformer module.
    """

    def __init__(self, channels, NUM):
        super(PT, self).__init__()
        
        self.NUM = NUM
        self.da = channels

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.fc1 = nn.Linear(128,channels) # 128 from output of input embedding layer
        
        self.fc_gamma = nn.Sequential(
            nn.Linear(channels,channels),
            nn.ReLU(),
            nn.Linear(channels,channels)
        )
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # fc1
        x_input = x # [B,da,N]
        x = self.fc1(x.permute(0,2,1)) # [B,N,da]
        x = x.permute(0,2,1) # [B,da,N]
        
        # compute query, key and value matrix
        # x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_q = self.q_conv(x)                   # [B, da, N] # No transpose !!
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]
         
        # find the central points
        x_q = x_q[:,:,0].unsqueeze(2).repeat(1,1,x.size(2)) # [B,da,N]

        # compute attention map and scale, the sorfmax
        # energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        # attention = self.softmax(energy)                      # [B, N, N]
        
        energy = self.fc_gamma(x_q.permute(0,2,1) - x_k.permute(0,2,1)) # [B,N,da]
        energy = energy.permute(0,2,1) / (math.sqrt(self.da)) # [B,da,N]
        attention = self.softmax(energy) # which dimension for softmax?
        
        # print('attention = ', attention.shape)

        # weighted sum
        # x_s = torch.bmm(x_v, attention)  # [B, de, N] 
        x_s = torch.mul(x_v, attention)  # [B, de, N] Hadamard product instead of dot-product !!
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # Visualize
        print('PT size = ', x_s.size())
        y_np = x_s.cpu()
        y_np = y_np.detach().numpy()
        plt.imshow(y_np[0,:,:])
        plt.savefig('visual/PT_fig_%s.png' %str(self.NUM))
        plt.show()
        
        # print('x_s = ', x_s.shape)
        # print('x_q = ', x_q.shape)
        # print('x_k = ', x_k.shape)
        # print('x_v = ', x_v.shape)
        # print('q_conv = ', self.q_conv)
        # print('k_conv = ', self.k_conv)
        # print('v_conv = ', self.v_conv)
        
        # residual
        x = x_s + x_input

        return x
