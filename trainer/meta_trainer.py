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

import argparse
import os,sys
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader.samplers import CategoriesSampler
from utils.misc import *
from utils.gpu_tools import occupy_memory
from tensorboardX import SummaryWriter
import tqdm
import time
import importlib
from gradcam import GradCAMpp
from score_map import ScoreCam
from rise import RISE
import captum
from captum.attr import (IntegratedGradients,NoiseTunnel)

class VizMod(torch.nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net = net
        self.layer4 = net.encoder.layer4
        self.features = net.encoder

    def forward(self,x,fast_weights,retFeat=False):

        feat = self.net.encoder(x,avgpool=False)

        featPooled = feat.mean(-1).mean(-1)

        

        featPooled = self.net.normalize_feature(featPooled)
        logits_q = self.net.base_learner(featPooled, fast_weights)

        if retFeat:
            return logits_q,feat 
        else:
            return logits_q

def loadDict(path,model,attention,nb_vec):

    sd = torch.load(path)['params']

    if attention == "br_npa" or attention == "bcnn":
        for key in sd:
            if key.find("base_learner.fc1_w") != -1 or key.find("base_learner.vars.0") != -1:
                if sd[key].shape != model.state_dict()[key].shape:
                    sd[key] = sd[key].repeat(1,nb_vec)
            elif key.find("hyperprior_combination_model.fc_w") != -1 or key.find("hyperprior_combination_model.hyperprior_mapping_vars.0") != -1:
                if sd[key].shape != model.state_dict()[key].shape:
                    sd[key] = sd[key].repeat(1,nb_vec)   
            elif key.find("hyperprior_basestep_model.fc_w") != -1 or key.find("hyperprior_basestep_model.hyperprior_mapping_vars.0") != -1:
                if sd[key].shape != model.state_dict()[key].shape:
                    sd[key] = sd[key].repeat(1,nb_vec)   

    miss,unexp = model.load_state_dict(sd,strict=False)

    print(miss[:10])
    print(unexp[:10])

    if attention == "cross":
        expToMiss = []
        actuallyMissing = []
        for key in miss:
            if key.find("base_learner.conv") != -1 or key.find("base_learner.vars") != -1 or key.find("conv1") != -1 or key.find("conv2") != -1:
                expToMiss.append(key)
            else:
                actuallyMissing.append(key)

        print("Missing",actuallyMissing,"but that is expected")

        if len(actuallyMissing) > 0:
            raise ValueError("Missing",actuallyMissing)

    elif attention == "bcnn":
        expToMiss, actuallyMissing = [],[]
        for key in miss:
            if key.find("base_learner.att.0") != -1 or key.find("base_learner.vars.1") != -1 or key.find("base_learner.vars.2") != -1:
                expToMiss.append(key)
            else:
                actuallyMissing.append(key)
        
        print("Missing",actuallyMissing,"but that is expected")

        if len(actuallyMissing) > 0:
            raise ValueError("Missing",actuallyMissing)
    else:
        if len(miss) > 0:
            raise ValueError("Missing keys",miss)

    if len(unexp) > 0:
        raise ValueError("Unexpected keys",unexp)

    return model

class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super().__init__(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot


class MetaTrainer(object):
    def __init__(self, args):
        self.args = args

        if args.dataset == 'miniimagenet':
            from dataloader.mini_imagenet import MiniImageNet as Dataset
            args.num_class = 64
            print('Using dataset: miniImageNet, base class num:', args.num_class)
        elif args.dataset == 'cub':
            from dataloader.cub import CUB as Dataset
            args.num_class = 100
            print('Using dataset: CUB, base class num:', args.num_class)
        elif args.dataset == 'tieredimagenet':
            from dataloader.tiered_imagenet import tieredImageNet as Dataset
            args.num_class = 351
            print('Using dataset: tieredImageNet, base class num:', args.num_class)
        elif args.dataset == 'fc100':
            from dataloader.fc100 import DatasetLoader as Dataset
            args.num_class = 60
            print('Using dataset: FC100, base class num:', args.num_class)
        elif args.dataset == 'cifar_fs':
            from dataloader.cifar_fs import DatasetLoader as Dataset
            args.num_class = 64
            print('Using dataset: CIFAR-FS, base class num:', args.num_class)
        else:
            raise ValueError('Please set the correct dataset.')

        self.Dataset = Dataset

        if args.mode == 'pre_train':
            print('Building pre-train model.')
            self.model = importlib.import_module('model.meta_model').MetaModel(args, dropout=args.dropout, mode='pre',highRes=args.dist)
        else:
            print('Building meta model.')
            self.model = importlib.import_module('model.meta_model').MetaModel(args, dropout=args.dropout, mode='meta',highRes=args.dist)

        if args.mode == 'pre_train':
            print ('Initialize the model for pre-train phase.')
        else:
            args.dir='pretrain_model/%s/%s/max_acc.pth'%(args.dataset,args.backbone)
            if not os.path.exists(args.dir):
                os.system('sh scripts/download_pretrain_model.sh')
            print ('Loading pre-trainrd model from:\n',args.dir)
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(args.dir,map_location="cuda" if torch.cuda.is_available() else "cpu")['params']
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
            for k,v in pretrained_dict.items():
                model_dict[k]=pretrained_dict[k]
            
            miss,unexp = self.model.load_state_dict(model_dict,strict=False)
            if len(miss) > 0 or len(unexp) > 0:
                print("DIR",args.dir)
                self.model = loadDict(args.dir,self.model,self.args.attention,self.args.nb_vec)

            if self.args.num_gpu>1:
                self.model = DataParallelModel(self.model)     
            self.model=self.model.cuda() if torch.cuda.is_available() else self.model

            if args.dist:
                oldAtt = args.attention
                args.attention = "none"
                self.teach_model = importlib.import_module('model.meta_model').MetaModel(args, dropout=args.dropout, mode='meta')
                self.teach_model = loadDict(args.dir,self.teach_model,self.args.attention,self.args.nb_vec)
                args.attention = oldAtt
                self.teach_model.eval()
        
                if self.args.num_gpu>1:
                    self.teach_model = DataParallelModel(self.teach_model)     
                self.teach_model=self.teach_model.cuda() if torch.cuda.is_available() else self.teach_model

        print('Building model finished.')

        if args.mode == 'pre_train':
            args.save_path = 'pre_train/%s-%s' % \
                         (args.dataset, args.backbone)  
        else:          
            args.save_path = 'meta_train/%s-%s-%s-%dway-%dshot' % \
                         (args.dataset, args.backbone, args.meta_update, args.way, args.shot)

        args.save_path=osp.join('logs', args.save_path)

        ensure_path(args.save_path)

        if args.mode == "meta_train":
            trainset = Dataset('train', args)
            if args.mode == 'pre_train':
                self.train_loader = DataLoader(dataset=trainset,batch_size=args.bs,shuffle=True, num_workers=args.num_workers, pin_memory=True)
            else:
                train_sampler = CategoriesSampler(trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
                self.train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

            valset = Dataset(args.set, args)
            val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
            self.val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

            val_loader=[x for x in self.val_loader]

            if args.mode == 'pre_train':
                self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': args.lr }, \
                                            {'params': self.model.fc.parameters(), 'lr': args.lr }], \
                                            momentum=0.9, nesterov=True, weight_decay=0.0005)
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
            else:
                if args.meta_update=='mtl':
                    new_para = filter(lambda p: p.requires_grad, self.model.encoder.parameters())
                else:
                    new_para = self.model.encoder.parameters()

                self.optimizer = torch.optim.SGD([{'params': new_para, 'lr': args.lr}, \
                    {'params': self.model.base_learner.parameters(), 'lr': self.args.lr}, \
                    {'params': self.model.get_hyperprior_combination_initialization_vars(), 'lr': self.args.lr_combination}, \
                    {'params': self.model.get_hyperprior_combination_mapping_vars(), 'lr': self.args.lr_combination_hyperprior}, \
                    {'params': self.model.get_hyperprior_basestep_initialization_vars(), 'lr': self.args.lr_basestep}, \
                    {'params': self.model.get_hyperprior_stepsize_mapping_vars(), 'lr': self.args.lr_basestep_hyperprior}], \
                    lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)

                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

        if self.args.ind_for_viz:
            self.vizMod = VizMod(self.model)

            if self.args.noise_tunnel:
                ig = IntegratedGradients(self.vizMod)
                self.nt = NoiseTunnel(ig)
                self.data_base = None
            elif self.args.rise:
                self.rise_mod = RISE(self.vizMod)
            else:
                model_dict = dict(type="resnet", arch=self.vizMod, layer_name='layer4', input_size=(448, 448))
                self.grad_cam = captum.attr.LayerGradCam(self.vizMod.forward,self.vizMod.layer4)
                self.grad_cam_pp = GradCAMpp(model_dict)
                self.guided_backprop_mod = captum.attr.GuidedBackprop(self.vizMod)
                self.score_mod = ScoreCam(self.vizMod)

    def save_model(self, name):

        name = '{}_{}'.format(name,self.args.model_id) if self.args.optuna else name

        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        args = self.args
        model = self.model
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        timer = Timer()
        global_count = 0
        writer = SummaryWriter(osp.join(args.save_path,'tf'))

        label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        label_shot = label_shot.to("cuda") if torch.cuda.is_available() else label_shot 
        label = label.to("cuda") if torch.cuda.is_available() else label

        for epoch in range(1, args.max_epoch + 1):
            start_time=time.time()

            tl = Averager()
            ta = Averager()

            tqdm_gen = tqdm.tqdm(self.train_loader)
            model.train()
            for i, batch in enumerate(tqdm_gen, 1):

                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.way 
                data_shot, data_query = data[:p], data[p:] 
                data_shot = data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1)
                
                if args.attention == "cross" and args.cross_att_loss:
                    logits,att_weights = self.model((data_shot, data_query),ytrain=label_shot,ytest=label)
                    loss1 = F.cross_entropy(logits, label)
                    loss2 = F.cross_entropy(att_weights, label)
                    loss = loss1 + args.cross_att_loss_weight * loss2

                else:
                    logits = model((data_shot, data_query)) 
                
                loss = F.cross_entropy(logits, label)

                if args.dist:
                    logits_teach = model((data_shot,data_query))
                    kl = F.kl_div(F.log_softmax(logits/args.kl_temp, dim=1),F.softmax(logits_teach/args.kl_temp, dim=1),reduction="batchmean")
                    loss = kl*args.kl_interp*args.kl_temp*args.kl_temp+loss*(1-args.kl_interp)

                acc = count_acc(logits, label)
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)

                total_loss = loss/args.bs
                writer.add_scalar('data/total_loss', float(total_loss), global_count)
                tqdm_gen.set_description('Epoch {}, Total loss={:.4f}, Acc={:.4f}.'
                    .format(epoch, total_loss.item(), acc))

                tl.add(total_loss.item())
                ta.add(acc)

                total_loss.backward()
                if i%args.bs==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            tl = tl.item()
            ta = ta.item()

            if epoch % 5 == 0:
                model.eval() 

                vl = Averager()
                va = Averager()

                tqdm_gen = tqdm.tqdm(self.val_loader)
                for i, batch in enumerate(tqdm_gen, 1):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]
                    p = args.shot * args.way
                    data_shot, data_query = data[:p], data[p:]
                    data_shot = data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1)
                    logits = model((data_shot, data_query))
                    
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)

                    vl.add(loss.item())
                    va.add(acc)
                    tqdm_gen.set_description('Episode {}: {:.2f}({:.2f})'.format(i, va.item() * 100, acc * 100))

                vl = vl.item()
                va = va.item()
                writer.add_scalar('data/val_loss', float(vl), epoch)
                writer.add_scalar('data/val_acc', float(va), epoch)

                print ('Validation acc:%.4f'%va)
                if va >= trlog['max_acc']:
                    print ('********* New best model!!! *********')
                    trlog['max_acc'] = va
                    trlog['max_acc_epoch'] = epoch
                    self.save_model("max_acc")

                trlog['val_loss'].append(vl)
                trlog['val_acc'].append(va)

                print('Best epoch {}, best val acc={:.4f}.'.format(trlog['max_acc_epoch'], trlog['max_acc']))

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)

            if args.dist:
                suff = "_{}".format(args.model_id)
            else:
                suff = ""

            torch.save(trlog, osp.join(args.save_path, 'trlog'+suff))
            if args.save_all:
                self.save_model('epoch-%d'%epoch)
                torch.save(self.optimizer.state_dict(), osp.join(args.save_path,'optimizer_latest.pth'))
            if epoch % 5 == 0:
                print ('This epoch takes %d seconds.'%(time.time()-start_time),'\nStill need %.2f hour to finish.'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
                self.lr_scheduler.step()

        writer.close()

    def normMap(self,maps):
        maps = 255*(maps-maps.min())/(maps.max()-maps.min())
        maps = maps.byte()
        return maps
    
    def computeViz(self,ind,data,targ,fast_weights):

        data,targ = data[ind:ind+1],targ[ind:ind+1]

        fast_weights = fast_weights[0].unsqueeze(0)

        inp = (data,fast_weights)

        if self.args.noise_tunnel:
            if self.data_base is None:
                self.data_base = torch.zeros_like(data)
                if data.device == "cuda":
                    self.data_base = self.data_base.cuda()

            attr_sq = self.nt.attribute(inp, nt_type='smoothgrad_sq', stdevs=0.02, nt_samples=16,nt_samples_batch_size=3, target=targ)
            attr_var = self.nt.attribute(inp, nt_type='vargrad', stdevs=0.02, nt_samples=16,nt_samples_batch_size=3, target=targ)
            return {"sq":attr_sq,"var":attr_var}

        elif self.args.rise:
            rise_map = self.rise_mod(data,fast_weights,targ).detach().cpu()
            return {"rise":rise_map}

        else:
            mask = self.grad_cam.attribute(inp,targ).detach().cpu()
            mask_pp = self.grad_cam_pp(data,fast_weights,targ).detach().cpu()
            map = self.guided_backprop_mod.attribute(inp,targ)[0].detach().cpu()
            score_map = self.score_mod.generate_cam(data,fast_weights,targ).detach().cpu()
            return {"gradcam":mask,"gradcam_pp":mask_pp,"guided":map,"score_map":score_map}

    def eval(self,finalTest=False):
        model = self.model
        args = self.args
        result_list=[args.save_path]

        if args.optuna:
            suff = "_{}".format(args.model_id)
        else:
            suff = ""

        if not os.path.exists(osp.join(args.save_path, 'trlog'+suff)):
            trlog = {}
            trlog['args'] = vars(args)
        else:
            trlog = torch.load(osp.join(args.save_path, 'trlog'+suff))

        torch.save(torch.get_rng_state(),"results/{}/rng_state_{}.pth".format(args.exp_id,args.model_id))

        if args.model_id != "dist":
            dist_rng_state = torch.load("results/{}/rng_state_dist.pth".format(args.exp_id))
            torch.set_rng_state(dist_rng_state)
  
        test_set = self.Dataset('val' if args.test_on_val else 'test')
        sampler = CategoriesSampler(test_set.label, 3000, args.way, args.shot + args.query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        test_acc_record = np.zeros((3000,))

        if self.args.optuna:
            name = "max_acc_{}".format(self.args.model_id)
        else:
            name = "max_acc"

        if finalTest and self.args.optuna:
            print("models/{}/model{}_best.pth".format(self.args.exp_id,self.args.model_id))
            model = loadDict("models/{}/model{}_best.pth".format(self.args.exp_id,self.args.model_id),model,self.args.attention,self.args.nb_vec)
        else:
            print(osp.join(args.save_path, name + '.pth'))
            model = loadDict(osp.join(args.save_path, name + '.pth'),model,self.args.attention,self.args.nb_vec)
        model.eval()

        ave_acc = Averager()
        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        if finalTest:
            allAtt,allNorm,allImgs = [],[],[]

        tqdm_gen = tqdm.tqdm(loader)

        inds = np.arange(args.way*args.query)

        if self.args.ind_for_viz and os.path.exists("results/{}/{}_vizDic.pth".format(self.args.exp_id,self.args.model_id)):
            vizDic= torch.load("results/{}/{}_vizDic.pth".format(self.args.exp_id,self.args.model_id))
        else:
            vizDic = {}

        for i, batch in enumerate(tqdm_gen, 1):

            if finalTest and self.args.ind_for_viz:
                matching_inds = list(set(self.args.ind_for_viz).intersection(set(list(inds+args.way*args.query*(i-1)))))
                forward = len(matching_inds) > 0 or not self.args.only_viz
            else:
                forward = True

            if forward:
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                k = args.way * args.shot
                data_shot, data_query = data[:k], data[k:]
                data_shot = data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1)
                ret = model((data_shot, data_query),retMaps=finalTest)
                logits = ret[0] if finalTest else ret

                acc = count_acc(logits, label)
                ave_acc.add(acc)
                test_acc_record[i-1] = acc
                tqdm_gen.set_description('Episode {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

            if finalTest and forward:
                
                norm = ret[1]

                norm = self.normMap(norm.cpu())
                allNorm.append(norm)
    
                data_unorm = self.normMap(data[k:].cpu())
                allImgs.append(data_unorm)

                if self.args.attention != "none":
                    attMaps = ret[2]
                    attMaps = self.normMap(attMaps.cpu())
                    allAtt.append(attMaps)

                    fast_weights = ret[3]
                else:
                    fast_weights = ret[2]

            if finalTest and self.args.ind_for_viz:
                if len(matching_inds) > 0:
                    for ind in matching_inds:
                        print("Ind",ind)
                        subDic = self.computeViz(ind%len(data_query),data_query,label,fast_weights)
                        if ind in vizDic:
                            vizDic[ind].update(subDic)
                        else:
                            vizDic[ind] = subDic
           
                        vizDic[ind]["imgs"] = data_query[ind%len(data_query)]

        m, pm = compute_confidence_interval(test_acc_record)

        if self.args.ind_for_viz:
            torch.save(vizDic,"results/{}/{}_vizDic.pth".format(self.args.exp_id,self.args.model_id))

        if not 'max_acc_epoch' in trlog:
            trlog['max_acc_epoch'] = -1
        if not "max_acc" in trlog:
            trlog["max_acc"] = -1

        result_list.append('Best validation epoch {},\nbest validation acc {:.4f}, \nbest test acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        result_list.append('Test acc {:.4f} + {:.4f}'.format(m, pm))
        print (result_list[-2])
        print (result_list[-1])
        save_list_to_txt(os.path.join(args.save_path,'results.txt'),result_list)

        if not os.path.exists("results/{}".format(args.exp_id)):
            os.makedirs("results/{}".format(args.exp_id))
        
        writeHeader = not os.path.exists("results/{}/{}.csv".format(args.exp_id,args.model_id)) 
      
        with open("results/{}/{}.csv".format(args.exp_id,args.model_id),"a") as file:
            if writeHeader:
                print("model_id,m,pm",file=file)

            print("{},{},{}".format(args.model_id,m,pm),file=file)

        if finalTest:
            suff = 'val' if args.test_on_val else 'test'

            if not self.args.ind_for_viz:
                allNorm = torch.cat(allNorm,dim=0)
                print("end")
                testNorm(allNorm.float().view(allNorm.shape[0],allNorm.shape[1],-1))
                np.save("./results/{}/norm_{}_{}.npy".format(self.args.exp_id,self.args.model_id,suff),allNorm.numpy())

            allImgs = torch.cat(allImgs,dim=0)
            np.save("./results/{}/imgs_{}_{}.npy".format(self.args.exp_id,self.args.model_id,suff),allImgs.numpy())

            if self.args.attention != "none":
                allAtt = torch.cat(allAtt,dim=0)
                np.save("./results/{}/attMaps_{}_{}.npy".format(self.args.exp_id,self.args.model_id,suff),allAtt.numpy())

        return m

    def pre_train(self):
        model = self.model
        args = self.args
        lr_scheduler = self.lr_scheduler
        optimizer = self.optimizer
        train_loader = self.train_loader
        val_loader = self.val_loader
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        timer = Timer()
        global_count = 0
        writer = SummaryWriter(osp.join(args.save_path,'tf'))

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        for epoch in range(1, args.max_epoch + 1):
            print (args.save_path)

            start_time=time.time()
            model = model.train()
            model.mode = 'pre'
            tl = Averager()
            ta = Averager()

            tqdm_gen = tqdm.tqdm(train_loader)
            for i, batch in enumerate(tqdm_gen, 1):

                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, train_label = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = model(data) 
                loss = F.cross_entropy(logits, train_label) 
                acc = count_acc(logits, train_label)

                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                total_loss = loss
                writer.add_scalar('data/total_loss', float(total_loss), global_count)
                tqdm_gen.set_description('Epoch {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
                tl.add(total_loss.item())
                ta.add(acc)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            tl = tl.item()
            ta = ta.item()

            model=model.eval()
            model.mode = 'meta'
            vl = Averager()
            va = Averager()

            if epoch < args.val_epoch:
                vl=0
                va=0
            else:
                tqdm_gen = tqdm.tqdm(val_loader)
                for i, batch in enumerate(tqdm_gen, 1):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]
                    p = args.shot * args.way
                    data_shot, data_query = data[:p], data[p:]  
                    data_shot = data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1)
                    logits = model.preval_forward(data_shot, data_query)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    vl.add(loss.item())
                    va.add(acc)

                vl = vl.item()
                va = va.item()
            writer.add_scalar('data/val_loss', float(vl), epoch)
            writer.add_scalar('data/val_acc', float(va), epoch)
            tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

            if va >= trlog['max_acc']:
                print ('********* New best model!!! *********')
                trlog['max_acc'] = va
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
                torch.save(optimizer.state_dict(), osp.join(args.save_path, 'optimizer_best.pth'))

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            torch.save(trlog, osp.join(args.save_path, 'trlog'))

            if args.save_all:

                self.save_model('epoch-%d'%epoch)
                torch.save(optimizer.state_dict(), osp.join(args.save_path,'optimizer_latest.pth'))

            print('Best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            print ('This epoch takes %d seconds'%(time.time()-start_time),'\nStill need %.2f hour to finish'%((time.time()-start_time)*(args.max_epoch-epoch)/3600))
            lr_scheduler.step()

        writer.close()
        result_list=['Best validation epoch {},\nbest val Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'],)]
        save_list_to_txt(os.path.join(args.save_path,'results.txt'),result_list)

def testNorm(norm):
    norm = norm.view(norm.shape[0],norm.shape[1],-1)
    norm_permuted = norm.permute(2,1,0)
    even_indexes = (torch.arange(norm_permuted.shape[0]) % 2 == 0)
    odd_indexes = ~even_indexes
    norm_even_permuted,norm_odd_permuted = norm_permuted[even_indexes],norm_permuted[odd_indexes]
    norm_even, norm_odd = norm_even_permuted.permute(2,1,0),norm_odd_permuted.permute(2,1,0)
    print(norm_even.shape,norm_odd.shape,norm_even.mean().item(),norm_even.max().item(),norm_odd.mean().item(),norm_odd.max().item())
