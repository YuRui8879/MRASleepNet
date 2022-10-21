import torch.nn as nn
import torch
from torch.nn import functional as F

class MRASleepNet(nn.Module):
    def __init__(self):
        super(MRASleepNet,self).__init__()
        self.fe = FE()
        self.mra = MRA()
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.convt = nn.Conv1d(1,64,4000,bias=False)
        self.bnt = nn.BatchNorm1d(64)
        self.gmlp1 = gMLPBlock(128,256,63)
        
        self.fc = nn.Sequential(
            nn.Linear(192,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,5),
            nn.Softmax(-1)
        )

    def forward(self,x):
        batch_size = x.size(0)
        x = x.view(x.size(0),1,x.size(-1))
        xt = self.bnt(self.convt(x)).view(batch_size,-1)
        x = self.fe(x)
        x = self.mra(x)
        x = x.view(x.size(0),x.size(2),x.size(1))
        x = self.gmlp1(x)
        x = x.reshape(x.size(0),x.size(2),x.size(1))
        x = self.gp(x).view(batch_size,-1)
        x = torch.cat((x,xt),1)
        x = self.fc(x)
        return x



class FE(nn.Module):
    def __init__(self):
        super(FE,self).__init__()
        self.conv1 = CNNBlock(1,64,49,2)
        self.conv2 = CNNBlock(64,128,7,2)
        self.conv3 = CNNBlock(128,128,7,2)
        self.maxpool1 = nn.MaxPool1d(2,2)
        self.maxpool2 = nn.MaxPool1d(4,4)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        return x

class CNNBlock(nn.Module):
    def __init__(self,in_ch,out_ch,filter_size,stride = 2):
        super(CNNBlock,self).__init__()
        self.conv1 = nn.Conv1d(in_ch,out_ch,filter_size,stride,filter_size//2,bias=False)
        self.conv2 = nn.Conv1d(out_ch,out_ch * 2,1,bias = False)
        self.conv3 = nn.Conv1d(out_ch * 2,out_ch,1,bias = False)
        self.convt = nn.Conv1d(in_ch,out_ch,1,stride,bias=False)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch * 2)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.bnt = nn.BatchNorm1d(out_ch)
        
    def forward(self,x):
        shortcut = self.bnt(self.convt(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += shortcut
        x = self.relu(x)
        return x

class MRA(nn.Module):
    def __init__(self):
        super(MRA,self).__init__()
        self.conv1 = nn.Conv1d(128,128,7,1,3)
        self.conv2 = nn.Conv1d(128,128,25,1,12)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(128,32,1,bias=False)
        self.fc1 = nn.Conv1d(32,128,1,bias=False)
        self.fc2 = nn.Conv1d(32,128,1,bias=False)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(-1)

    def forward(self,x):
        ch1 = self.bn1(self.conv1(x))
        ch2 = self.bn2(self.conv2(x))
        x = ch1 + ch2
        x = self.avgpool(x)
        x = self.relu(self.fc(x))
        wch1 = self.fc1(x)
        wch2 = self.fc2(x)
        attn = self.softmax(torch.cat((wch1,wch2),-1))
        x1 = ch1 * attn[:,:,0].unsqueeze(-1)
        x2 = ch2  * attn[:,:,1].unsqueeze(-1)
        x = x1 + x2
        return x


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
 
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
 
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out

if __name__ == '__main__':
    x = torch.rand(2,4000)
    model = MRASleepNet()
    y = model(x)
    print(y.shape)
