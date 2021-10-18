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

import argparse,os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader.samplers import CategoriesSampler
from utils.misc import *
from utils.gpu_tools import occupy_memory, set_gpu
from tensorboardX import SummaryWriter
import tqdm
import time
import importlib
from trainer.meta_trainer import MetaTrainer
import optuna,sqlite3
import shutil

    
def getBestTrial(exp_id,model_id,trialNb=None):
    con = sqlite3.connect("./results/{}/{}_hypSearch.db".format(exp_id,model_id))
    curr = con.cursor()

    curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
    query_res = curr.fetchall()

    query_res = list(filter(lambda x:not x[1] is None,query_res))

    trialIds = [id_value[0] for id_value in query_res]
    values = [id_value[1] for id_value in query_res]

    if not trialNb is None:
        trialIds = trialIds[:trialNb]
        values = values[:trialNb]

    bestTrial = trialIds[np.array(values).argmax()]

    return bestTrial

def setBestParams(args):

    trialId = getBestTrial(args.exp_id,args.model_id)

    con = sqlite3.connect("./results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
    curr = con.cursor()

    dic = vars(args)

    params = ["lr","temperature","step_size","gamma","dropout","base_lr","kl_temp","kl_interp","lr_combination",\
              "lr_combination_hyperprior","lr_basestep","lr_basestep_hyperprior"]

    for param in params:
        curr.execute("SELECT param_value FROM trial_params WHERE trial_id == {} and param_name == '{}' ".format(trialId,param))
        query_res = curr.fetchall()
        if len(query_res) > 0:
            param_val = type(dic[param])(query_res[0][0])
            args.__dict__[param] = param_val

    return args

def run(args,trial=None,finalTest=False):

    if args.optuna and not finalTest:
        args.lr = trial.suggest_float("lr",0.0005, 0.05, log=True)
        args.temperature = trial.suggest_float("temperature",4, 12, step=4)
        args.step_size = trial.suggest_int("step_size",5,20,step=5)
        args.gamma = trial.suggest_float("gamma",0.3, 0.7, step=0.2)
        args.dropout = trial.suggest_float("dropout",0.3, 0.7, step=0.2)
        args.base_lr = trial.suggest_float("base_lr",0.01, 0.5, log=True)
        if args.dist:
            args.kl_temp = trial.suggest_float("kl_temp", 1, 21, step=5)
            args.kl_interp = trial.suggest_float("kl_interp", 0.1, 1, step=0.1)

        if args.more_params:
            args.lr_combination = trial.suggest_float("lr_combination",1e-7, 1e-5, log=True)
            args.lr_combination_hyperprior = trial.suggest_float("lr_combination_hyperprior",1e-7, 1e-5, log=True)
            args.lr_basestep = trial.suggest_float("lr_basestep",1e-7, 1e-5, log=True)
            args.lr_basestep_hyperprior = trial.suggest_float("lr_basestep_hyperprior",1e-7, 1e-5, log=True)

    trainer = MetaTrainer(args)
    if args.mode == 'meta_train':
        if finalTest and not os.path.exists("models/{}/model{}_best.pth".format(args.exp_id,args.model_id)):
            print('Start meta-train phase.')
            trainer.train()
            name = "max_acc_"+args.model_id
            path = osp.join(args.save_path, name + '.pth')
            shutil.copyfile(path,"models/{}/model{}_best.pth".format(args.exp_id,args.model_id))
        print('Start meta-test phase.')
        value = trainer.eval(finalTest=finalTest)
    elif args.mode == 'meta_eval':
        print('Start meta-test phase.')
        value = trainer.eval(finalTest=finalTest)
    elif args.mode == 'pre_train':
        print('Start pre-train phase.')
        trainer.pre_train()

    return value 

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'tieredimagenet', 'fc100'])
parser.add_argument('-datadir', type=str, default=None)
parser.add_argument('-set',type=str,default='val',choices=['test','val'])
parser.add_argument('-mode',type=str,default='meta_train',choices=['pre_train', 'meta_train', 'meta_eval'])
parser.add_argument('-bs', type=int, default=1,help='batch size')
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-temperature', type=float, default=8)
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-val_frequency',type=int,default=100)
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=5)
parser.add_argument('-query', type=int, default=16)
parser.add_argument('-val_episode', type=int, default=3000)
parser.add_argument('-val_epoch', type=int, default=40)
parser.add_argument('-backbone', type=str, default='resnet12', choices=['wrn', 'resnet12'])
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-save_all',action='store_true',help='save models on each epoch')
parser.add_argument('-meta_update',type=str,default='mtl',choices=['ft','mtl'])
parser.add_argument('--hyperprior_init_mode', type=str, default='LAS', choices=['LAS', 'EQU'])
parser.add_argument('--hyperprior_combination_softweight', type=float, default=1e-4)
parser.add_argument('--hyperprior_basestep_softweight', type=float, default=1e-4)
parser.add_argument('-base_init',type=str,default='feature',choices=['feature'])
parser.add_argument('-base_epoch', type=int, default=100)
parser.add_argument('-base_lr', type=float, default=0.1)
parser.add_argument('-base_lr_encoder', type=float, default=0.01)
parser.add_argument('-lr_combination', type=float, default=1e-6)
parser.add_argument('-lr_combination_hyperprior', type=float, default=1e-6)
parser.add_argument('-lr_basestep', type=float, default=1e-6)
parser.add_argument('-lr_basestep_hyperprior', type=float, default=1e-6)
parser.add_argument('-kl_temp', type=float, default=8)
parser.add_argument('-kl_interp', type=float, default=0.5)
parser.add_argument('-sleep',type=float,help='hour',default=0.0)
parser.add_argument('-gpu', default='0')
parser.add_argument('-gpu_occupy',action='store_true')
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-num_workers', type=int, default=8)
parser.add_argument('-attention',type=str,default='none')
parser.add_argument('-nb_vec',type=int,default=3)
parser.add_argument('-dist',action="store_true")
parser.add_argument('-optuna',action="store_true")
parser.add_argument('-more_params',action="store_true")
parser.add_argument('-optuna_trial_nb',type=int,default=20)

parser.add_argument('-model_id',type=str,default='default')
parser.add_argument('-exp_id',type=str,default='default')

parser.add_argument('-ind_for_viz',type=int,nargs="*")
parser.add_argument('-only_viz',action="store_true")
parser.add_argument('-noise_tunnel',action="store_true")
parser.add_argument('-rise',action="store_true")


args = parser.parse_args()
print(vars(args))

if args.seed==0:
    print ('Random mode.')
    torch.backends.cudnn.benchmark = True
else:
    import random
    print ('Fixed random seed:', args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_gpu=set_gpu(args)
args.num_gpu=num_gpu
if args.gpu_occupy:
    occupy_memory(args.gpu)
    print('Occupy GPU memory in advance.')

if args.optuna:
    
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/{}/".format(args.exp_id)):
        os.makedirs("results/{}/".format(args.exp_id))

    def objective(trial):
        return run(args,trial=trial)

    study = optuna.create_study(direction="maximize",\
                                storage="sqlite:///./results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id), \
                                study_name=args.model_id,load_if_exists=True)

    con = sqlite3.connect("./results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
    curr = con.cursor()

    failedTrials = 0
    for elem in curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall():
        if elem[1] is None:
            failedTrials += 1

    trialsAlreadyDone = len(curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall())

    if trialsAlreadyDone-failedTrials < args.optuna_trial_nb:

        studyDone = False
        while not studyDone:
            print("N trials left",args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
            study.optimize(objective,n_trials=args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
            studyDone = True

    args = setBestParams(args)
    run(args,finalTest=True)

else:
    run(args,finalTest=True)

