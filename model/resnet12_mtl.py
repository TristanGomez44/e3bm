#   Copyright (c) 2020 Yaoyao Liu. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

from warnings import simplefilter
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.conv2d_mtl import Conv2dMtl

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet(nn.Module):

    def __init__(self,block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5,attention="none",nb_vec=3,highRes=False):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.attention = attention
        self.nb_vec = nb_vec

        cfg = [64,160, 320, 640]

        self.layer1 = self._make_layer(block, cfg[0], stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, cfg[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, cfg[2], stride=1 if highRes else 2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, cfg[3], stride=1 if highRes else 2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        #if avg_pool:
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, Conv2dMtl):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dMtl(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x,retMaps=False,avgpool=True):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        if retMaps:
            norm = torch.sqrt(torch.pow(x,2).sum(dim=1,keepdim=True))

        if self.attention == "br_npa":
            x,attMaps = representativeVectors(x,self.nb_vec)

        elif self.attention == "none" and avgpool:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)
        
        if retMaps and self.attention == "br_npa":
            return x,attMaps,norm 
        elif retMaps:
            return x,norm
        else:
            return x

def representativeVectors(x,nbVec=3):

    xOrigShape = x.size()

    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    raw_reprVec_score = norm.clone()

    repreVecList = []
    simList = []
    for _ in range(nbVec):
        _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)
        simNorm = sim/sim.sum(dim=1,keepdim=True)
        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)
        repreVecList.append(reprVec)
        raw_reprVec_score = (1-sim)*raw_reprVec_score
        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])
        simList.append(simReshaped)
    
    repreVecList = torch.cat(repreVecList,dim=-1)
    simList = torch.cat(simList,dim=1)    

    return repreVecList,simList
