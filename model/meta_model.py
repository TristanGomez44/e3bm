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

from numpy.lib.shape_base import vsplit
import  torch
import torch.nn as nn
from utils.misc import euclidean_metric
import torch.nn.functional as F
from model.conv2d_mtl import Conv2dMtl
import math 
import sys


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_reg(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)

        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out
    
class BaseLearner(nn.Module):
    def __init__(self, args, z_dim,attention="none",nb_vec=3):
        super().__init__()
        self.args = args
        self.z_dim = z_dim*(args.nb_vec if args.attention=="br_npa" or args.attention=="bcnn" else 1)
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)

        cfg = [64,160, 320, 640]
        self.cfg = cfg

        self.nbVec = nb_vec
        self.attention = attention
        if self.attention == "bcnn":
            attention = []
            attention.append(conv3x3_reg(cfg[-1], self.nbVec))
            attention.append(nn.ReLU())
            self.att = nn.Sequential(*attention)
            self.vars.extend(list(self.att.parameters()))

            self.avgpool = nn.AdaptiveAvgPool2d(1)

    def compAtt(self,x,the_vars=None):

        if the_vars is None:
            the_vars = self.vars

        weight,bias = the_vars[1:]
        attMaps = F.relu(F.conv2d(x,weight,bias,padding=1))
        
        x = (attMaps.unsqueeze(2)*x.unsqueeze(1)).reshape(x.size(0),x.size(1)*(attMaps.size(1)),x.size(2),x.size(3))
        x = self.avgpool(x)
        return x.view(x.size(0), -1),attMaps

    def get_attention(self, a,quer=True,the_vars=None):
        input_a = a

        if the_vars is None:
            the_vars = self.vars

        w1,b1,w2,b2 = the_vars[1:]

        a = a.mean(3) 
        a = a.transpose(1, 3) 
        a = F.relu(F.conv2d(a,w1,b1))
        a = F.conv2d(a,w2,b2) 
        a = a.transpose(1, 3)
        a = a.unsqueeze(3) 
        
        a = torch.mean(input_a * a, -1) 
        
        if quer:
            att_weights = torch.softmax(a.mean(dim=-1,keepdim=True),dim=1).unsqueeze(-1)
        else:
            att_weights = torch.softmax(a.mean(dim=-1,keepdim=True),dim=2).unsqueeze(-1)

        a = F.softmax(a / 0.025, dim=-1) + 1

        return a,att_weights

    def poolFeat(self,input_x,the_vars=None,retMaps=False):
        if self.attention == "bcnn":
            if retMaps:
                norm = torch.sqrt(torch.pow(input_x,2).sum(dim=3))

            input_x,attMaps = self.compAtt(input_x,the_vars)
            
            if retMaps:
                return input_x,attMaps,norm
            else:
                return input_x

        elif self.attention == "cross":
            input_x=F.adaptive_avg_pool2d(input_x,1).squeeze(-1).squeeze(-1)

            return input_x

    def forward(self, input_x, the_vars=None,retMaps=False):

        if the_vars is None:
            the_vars = self.vars

        ret = self.poolFeat(input_x,the_vars,retMaps=retMaps)
        input_x = ret[0] if retMaps else ret

        fc1_w = the_vars[0]

        net = F.linear(F.normalize(input_x, p=2, dim=1), F.normalize(fc1_w, p=2, dim=1))
        
        if retMaps and self.attention == "bcnn":
            return net,ret[1],ret[2]
        else:
            return net

    def parameters(self):
        return self.vars

class HyperpriorCombination(nn.Module):
    def __init__(self, args, update_step, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        if args.hyperprior_init_mode=='LAS':
            for idx in range(update_step-1):
                self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([0.0])))
            self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([1.0])))
        else:
            for idx in range(update_step):
                self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([1.0/update_step])))

        self.hyperprior_mapping_vars = nn.ParameterList()
        mult = args.nb_vec if args.attention == "br_npa" or args.attention == "bcnn" else 1
        self.fc_w = nn.Parameter(torch.ones([update_step, z_dim*2*mult]))
        torch.nn.init.kaiming_normal_(self.fc_w)
        self.hyperprior_mapping_vars.append(self.fc_w)
        self.fc_b = nn.Parameter(torch.zeros(update_step))
        self.hyperprior_mapping_vars.append(self.fc_b)
        self.hyperprior_softweight = args.hyperprior_combination_softweight

    def forward(self, input_x, grad, step_idx):
        mean_x = input_x.mean(dim=0)
        mean_grad = grad[0].mean(dim=0)
        net = torch.cat((mean_x, mean_grad), 0)
        net = F.linear(net, self.fc_w, self.fc_b)
        net = net[step_idx]
        net = self.hyperprior_initialization_vars[step_idx] + self.hyperprior_softweight*net
        return net

    def get_hyperprior_initialization_vars(self):
        return self.hyperprior_initialization_vars

    def get_hyperprior_mapping_vars(self):
        return self.hyperprior_mapping_vars

class HyperpriorBasestep(nn.Module):
    def __init__(self, args, update_step, update_lr, z_dim):
        super().__init__()
        self.args = args
        self.hyperprior_initialization_vars = nn.ParameterList()
        for idx in range(update_step):
            self.hyperprior_initialization_vars.append(nn.Parameter(torch.FloatTensor([update_lr])))

        self.hyperprior_mapping_vars = nn.ParameterList()
        mult = args.nb_vec if args.attention == "br_npa" or args.attention == "bcnn" else 1
        self.fc_w = nn.Parameter(torch.ones([update_step, z_dim*2*mult]))
        torch.nn.init.kaiming_normal_(self.fc_w)
        self.hyperprior_mapping_vars.append(self.fc_w)
        self.fc_b = nn.Parameter(torch.zeros(update_step))
        self.hyperprior_mapping_vars.append(self.fc_b)
        self.hyperprior_softweight = args.hyperprior_basestep_softweight


    def forward(self, input_x, grad, step_idx):
        mean_x = input_x.mean(dim=0)
        mean_grad = grad[0].mean(dim=0)
        net = torch.cat((mean_x, mean_grad), 0)
        net = F.linear(net, self.fc_w, self.fc_b)
        net = net[step_idx]
        net = self.hyperprior_initialization_vars[step_idx] + self.hyperprior_softweight*net
        return net

    def get_hyperprior_initialization_vars(self):
        return self.hyperprior_initialization_vars

    def get_hyperprior_mapping_vars(self):
        return self.hyperprior_mapping_vars

class MetaModel(nn.Module):
    def __init__(self, args, dropout=0.2, mode='meta',highRes=False):
        super().__init__()
        self.args = args
        self.mode = mode

        self.init_backbone(highRes)
        self.base_learner = BaseLearner(args, self.z_dim,attention=self.args.attention,nb_vec=self.args.nb_vec)
        self.update_lr = self.args.base_lr
        self.update_step = self.args.base_epoch

        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            self.label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            self.label_shot = label_shot.type(torch.LongTensor)

        if self.mode == 'meta':
            self.hyperprior_combination_model = HyperpriorCombination(args, self.update_step, self.z_dim)
            self.hyperprior_basestep_model = HyperpriorBasestep(args, self.update_step, self.update_lr, self.z_dim)

        if self.args.attention == "cross":
            self.conv1 = nn.Conv1d(25, 5, 1)
            self.conv2 = nn.Conv1d(5, 25, 1, stride=1, padding=0)

    def init_backbone(self,highRes):
        if self.args.backbone == 'resnet12':
            if self.mode == 'pre':
                from model.resnet12 import ResNet
            else:
                if self.args.meta_update=='mtl':
                    from model.resnet12_mtl import ResNet
                else:
                    from model.resnet12 import ResNet
            self.encoder = ResNet(attention=self.args.attention,nb_vec=self.args.nb_vec,highRes=highRes)
            self.z_dim = 640
        elif self.args.backbone == 'wrn':
            if self.mode == 'pre':
                from Models.backbone.wrn import ResNet
            else:
                if self.args.meta_update=='mtl':
                    from model.wrn_mtl import ResNet
                else:
                    from model.wrn import ResNet
            self.encoder = ResNet()
            self.z_dim = 640
        else:
            raise ValueError('Please set the correct backbone')

        if self.mode == 'pre':
            self.fc = nn.Sequential(nn.Linear(self.z_dim, self.args.num_class))

    def forward(self, inputs,retMaps=False):
        if self.mode=='pre':
            return self.pretrain_forward(inputs)
        elif self.mode=='meta':
            data_shot, data_query = inputs
            return self.meta_forward(data_shot, data_query,retMaps=retMaps)
        else:
            raise ValueError('Please set the correct mode')

    def pretrain_forward(self, input):
        return self.fc(self.encoder(input))

    def normalize_feature(self, x):
        x = x-x.mean(-1).unsqueeze(-1)
        return x

    def fusion(self, embedding):
        embedding = embedding.view(self.args.shot, self.args.way, -1)
        embedding = embedding.mean(0)
        return embedding

    def get_hyperprior_combination_initialization_vars(self):
        return self.hyperprior_combination_model.get_hyperprior_initialization_vars()

    def get_hyperprior_basestep_initialization_vars(self):
        return self.hyperprior_basestep_model.get_hyperprior_initialization_vars()

    def get_hyperprior_combination_mapping_vars(self):
        return self.hyperprior_combination_model.get_hyperprior_mapping_vars()

    def get_hyperprior_stepsize_mapping_vars(self):
        return self.hyperprior_basestep_model.get_hyperprior_mapping_vars()

    def cam(self, f1, f2):

        f1 = f1.view(self.args.way*self.args.shot,f1.shape[1],f1.shape[2]*f1.shape[3])
        f2 = f2.view(self.args.way*self.args.query,f2.shape[1],f2.shape[2]*f2.shape[3])
        #f1 25 640 25
        #f2 80 640 25

        f1 = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2 = F.normalize(f2, p=2, dim=2, eps=1e-12)

        r = (f1.unsqueeze(0).unsqueeze(3)*f2.unsqueeze(1).unsqueeze(4)).sum(dim=2)

        r = r.view(f2.shape[0]*f1.shape[0],r.shape[2],r.shape[3])
    
        ##############  attention for f1
        #r 80*25 25 25
        w_f1 = self.conv2(torch.relu(self.conv1(r.mean(dim=-1,keepdim=True))))
        #w 80*25 25 1
        w_f1 = w_f1.view(f2.shape[0],f1.shape[0],w_f1.shape[1],1)
        #w 80 25 25 1

        a_f1_spat = torch.softmax(w_f1,dim=2)
        #w 80 25 25 1
        a_f1_spat =  a_f1_spat.permute(0,1,3,2)
        #w 80 25 1 25

        f1_quer = f1.unsqueeze(0)*a_f1_spat
        #f1_quer 80 25 640 25

        a_f1_quer = torch.softmax(w_f1.mean(dim=2,keepdim=True),dim=0)
        #w 80 25 1 1

        f1 = (f1_quer*a_f1_quer).sum(dim=0)
        #25 640 25
    
        ##############  attention for f1 
        #r 80*25 25 25
        r = r.permute(0,2,1)
        w_f2 = self.conv2(torch.relu(self.conv1(r.mean(dim=-1,keepdim=True))))
        #w 80*25 25 1
        w_f2 = w_f2.view(f2.shape[0],f1.shape[0],w_f2.shape[1],1)
        #w 80 25 25 1      

        a_f2_spat = torch.softmax(w_f2,dim=2)
        #w 80 25 25 1
        a_f2_spat =  a_f2_spat.permute(0,1,3,2)
        #w 80 25 1 25

        f2_shot = f2.unsqueeze(1)*a_f2_spat
        #f2_shot 80 25 640 25

        a_f2_shot = torch.softmax(w_f2.mean(dim=2,keepdim=True),dim=1)
        #w 80 25 1 1

        f2 = (f2_shot*a_f2_shot).sum(dim=1)
        #80 640 25

        attMaps = (a_f2_shot*a_f2_spat).sum(dim=1)

        #Choosing the top nb_vec attention maps
        _,top_inds = a_f2_shot.squeeze(-1).squeeze(-1).topk(self.args.nb_vec,dim=1)

        attMaps = []
        for i in range(len(top_inds)):
            attMaps.append(a_f2_spat[i][top_inds[i]].unsqueeze(0))
        attMaps = torch.cat(attMaps,dim=0)

        #80 1 1 25
        map_size = int(math.sqrt(attMaps.shape[-1]))
        attMaps = attMaps.view(attMaps.shape[0],self.args.nb_vec,map_size,map_size)
        #80 1 5 5

        f1 = f1.view(f1.shape[0],f1.shape[1],map_size,map_size)
        f2 = f2.view(f2.shape[0],f2.shape[1],map_size,map_size)
           
        return f1, f2,attMaps

    def meta_forward(self, data_shot, data_query,retMaps=False):
        data_query=data_query.squeeze(0)
        data_shot = data_shot.squeeze(0)

        ret = self.encoder(data_query,retMaps=retMaps and (self.args.attention == "br_npa" or self.args.attention == "none"))

        if retMaps and self.args.attention == "br_npa":
            embedding_query,attMaps,norm = ret
        elif retMaps and self.args.attention == "none":
            embedding_query,norm = ret
        else:
            embedding_query = ret

        embedding_shot = self.encoder(data_shot)
        embedding_shot = self.normalize_feature(embedding_shot)
        embedding_query = self.normalize_feature(embedding_query)

        if self.args.attention == "cross":
            embedding_shot,embedding_query,attMaps = self.cam(embedding_shot,embedding_query)
            emb_mean = embedding_shot.mean(dim=-1).mean(dim=-1)

            if retMaps:
                norm = torch.sqrt(torch.pow(embedding_query,2).sum(dim=1,keepdim=True))

        else:
            attMap_shot,attMap_quer = None,None
            if not self.args.attention == "bcnn":
                emb_mean = embedding_shot

        with torch.no_grad():
            if self.args.attention == "bcnn":
                embedding_shot_pooled = self.base_learner.poolFeat(embedding_shot)
            else:
                embedding_shot_pooled = emb_mean

            if self.args.shot==1:
                proto = embedding_shot_pooled.contiguous()
            else:
                proto= self.fusion(embedding_shot_pooled.contiguous())

            self.base_learner.fc1_w.data = proto

        fast_weights = self.base_learner.vars

        combination_value_list = []
        basestep_value_list = []
        batch_shot = embedding_shot
        batch_label = self.label_shot
        logits_q = self.base_learner(embedding_query, fast_weights)
        total_logits = 0.0 * logits_q

        retMaps_bcnn = retMaps and self.args.attention == "bcnn"

        for k in range(0, self.update_step):
            batch_shot = embedding_shot
            batch_label = self.label_shot
            logits = self.base_learner(batch_shot, fast_weights) * self.args.temperature
            loss = F.cross_entropy(logits, batch_label)
            grad = torch.autograd.grad(loss, fast_weights)
            generated_combination_weights = self.hyperprior_combination_model(embedding_shot_pooled, grad, k)
            generated_basestep_weights = self.hyperprior_basestep_model(embedding_shot_pooled, grad, k)
            fast_weights = list(map(lambda p: p[1] - generated_basestep_weights * p[0], zip(grad, fast_weights)))
            ret = self.base_learner(embedding_query, fast_weights,k == self.update_step -1 and retMaps_bcnn)
            if k == self.update_step -1 and retMaps_bcnn:
                attMaps,norm = ret[1],ret[2]
            
            logits_q = ret[0] if k == self.update_step -1 and retMaps_bcnn else ret
            logits_q = logits_q * self.args.temperature
            total_logits += generated_combination_weights * logits_q
            combination_value_list.append(generated_combination_weights)
            basestep_value_list.append(generated_basestep_weights)  

        if retMaps and self.args.attention != "none":
            return total_logits,norm,attMaps,fast_weights
        elif retMaps:
            return total_logits,norm,fast_weights
        else:
            return total_logits

    def preval_forward(self, data_shot, data_query):
        data_query=data_query.squeeze(0)
        data_shot = data_shot.squeeze(0)

        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        embedding_shot = self.normalize_feature(embedding_shot)
        embedding_query = self.normalize_feature(embedding_query)

        with torch.no_grad():
            if self.args.shot==1:
                proto = embedding_shot
            else:
                proto=self.fusion(embedding_shot)
            self.base_learner.fc1_w.data = proto

        fast_weights = self.base_learner.vars

        batch_shot = embedding_shot
        batch_label = self.label_shot
        logits_q = self.base_learner(embedding_query, fast_weights)
        total_logits = 0.0 * logits_q

        for k in range(0, self.update_step):

            batch_shot = embedding_shot
            batch_label = self.label_shot
            logits = self.base_learner(batch_shot, fast_weights) * self.args.temperature
            loss = F.cross_entropy(logits, batch_label)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.1 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
            logits_q = logits_q * self.args.temperature
            total_logits += logits_q

        return total_logits