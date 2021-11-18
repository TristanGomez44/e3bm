import glob
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import sys
import os,sys

def normMap(map,onlyMax=False):
    map_min = map.min(axis=-1,keepdims=True).min(axis=-2,keepdims=True).min(axis=-3,keepdims=True)
    map_max = map.max(axis=-1,keepdims=True).max(axis=-2,keepdims=True).max(axis=-3,keepdims=True)

    if not onlyMax:
        map = (map-map_min)/(map_max-map_min)
    else:
        map = map/map_max
    return map

def reshape(tens,smooth=True):

    if smooth:
        tens = torch.nn.functional.interpolate(tens,(300,300),mode="bilinear",align_corners=False)
    else:
        tens = torch.nn.functional.interpolate(tens,(300,300),mode="nearest")

    return tens

def applyCMap(tens,cmPlasma):
    tens = torch.tensor(cmPlasma(tens[0,0].numpy())[:,:,:3]).float()
    tens = tens.permute(2,0,1).unsqueeze(0)
    return tens

def loadNorm(exp_id,model_id,suff):
    norms = np.load("results/{}/norm_{}_{}.npy".format(exp_id,model_id,suff),mmap_mode="r")
    norms = normMap(norms,onlyMax=False)
    return norms

def loadAttNormImgs(exp_id,model_id,test_on_val=False):

    suff = "val" if test_on_val else "test"

    attMaps = np.load("results/{}/attMaps_{}_{}.npy".format(exp_id,model_id,suff),mmap_mode="r")

    if model_id.find("cross") != -1:
        #attMaps = attMaps[:,0:1]

        #3000 25 65 25
        attMaps = attMaps.transpose(0,2,1,3)
        #3000 65 25 25
        attMaps = attMaps.reshape(attMaps.shape[0]*attMaps.shape[1],attMaps.shape[2],attMaps.shape[3])
        #3000*65 25 25
        mapSize = int(math.sqrt(attMaps.shape[2]))
        attMaps = attMaps.reshape(attMaps.shape[0],attMaps.shape[1],mapSize,mapSize)
        #3000*65 25 5 5
        attMaps = attMaps.mean(axis=1,keepdims=True)
        #3000*65 1 5 5

    attMaps = normMap(attMaps)

    print("results/{}/attMaps_{}_{}.npy".format(exp_id,model_id,suff),attMaps.shape)
    
    imgs = np.load("results/{}/imgs_{}_{}.npy".format(exp_id,model_id,suff),mmap_mode="r")
    imgs = normMap(imgs)

    norms = loadNorm(exp_id,model_id,suff)

    attMaps = attMaps*norms

    return attMaps,norms,imgs

def mixAndCat(catImg,map,img):
    mix = 0.8*map+0.2*img.mean(dim=1,keepdims=True)
    return torch.cat((catImg,mix),dim=0)

def visMaps(exp_id,model_id,viz_id,vizOrder,cross_id,bcnn_id,nrows,img_to_plot,plot_id,test_on_val,only_dist):

    suff = "val" if test_on_val else "test"

    cmPlasma = plt.get_cmap('plasma')

    attMaps_cross,_,_ = loadAttNormImgs(exp_id,cross_id,test_on_val=test_on_val)

    attMaps,_,imgs = loadAttNormImgs(exp_id,model_id,test_on_val=test_on_val)

    attMaps_bcnn,_,_ = loadAttNormImgs(exp_id,bcnn_id,test_on_val=test_on_val)

    if not only_dist:
        norms = loadNorm(exp_id,viz_id,suff)
        viz_dic = torch.load(os.path.join("results",exp_id,viz_id+"_vizDic.pth"))

    catImg = None

    if len(img_to_plot) == 0:
        img_to_plot = np.arange(len(imgs))

    for i in img_to_plot:

        img = reshape(torch.tensor(imgs[i:i+1]))

        if catImg is None:
            catImg = img
        else:
            catImg = torch.cat((catImg,img),dim=0) 
        
        if not only_dist:
            if i in viz_dic:
                for key in vizOrder:

                    if key == "norm":
                        viz_map = torch.tensor(norms[i:i+1])  
                    else:
                        viz_map = viz_dic[i][key]

                    if key in ["sq","var"]:
                        if type(viz_map) is tuple:
                            viz_map = viz_map[0]

                    if key == "guided":
                        viz_map = torch.abs(viz_map)

                    if key in ["sq","var","guided"]:
                        viz_map = viz_map.mean(dim=1,keepdim=True)

                    viz_map = (viz_map-viz_map.min())/(viz_map.max()-viz_map.min())

                    if len(viz_map.shape) == 3:
                        viz_map = viz_map.unsqueeze(0)

                    if viz_map.shape[1] == 1:
                        viz_map = applyCMap(viz_map.cpu(),cmPlasma)

                    viz_map = reshape(viz_map,smooth=True)         
                    catImg = mixAndCat(catImg,viz_map.cpu(),img)

                #match(viz_dic[i]["imgs"],imgs,i)

            else:
                raise ValueError("Can't find {} in vizDic of {}".format(i,viz_id))

        catImg = preproc_and_cat(attMaps_bcnn,i,catImg,img,smooth=True)
        catImg = preproc_and_cat(attMaps_cross,i,catImg,img,smooth=True,cm=cmPlasma)
        catImg = preproc_and_cat(attMaps,i,catImg,img,smooth=True)

        if i % 80 == 79:
            
            outPath = "./vis/{}/{}_{}_{}_attMaps_{}.png".format(exp_id,model_id,i,plot_id,suff)
            torchvision.utils.save_image(catImg,outPath,nrow=nrows)
            catImg = None

    if not img_to_plot is None:
        outPath = "./vis/{}/{}_{}_attMaps_{}.png".format(exp_id,model_id,plot_id,suff)
        torchvision.utils.save_image(catImg,outPath,nrow=nrows)
        os.system("convert -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))

def preproc_and_cat(attMaps,i,catImg,img,smooth=False,cm=None):
    attMap = torch.tensor(attMaps[i:i+1])

    if not cm is None:
       attMap = applyCMap(attMap,cm) 

    attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())
    attMap = reshape(attMap,smooth=smooth)
    catImg = mixAndCat(catImg,attMap,img)
    return catImg

def match(img,imgs,ind):

    img = img.unsqueeze(0).to("cpu")
    img = img.view(img.shape[0],-1)
    imgs = torch.tensor(imgs).to("cpu")
    imgs = imgs.view(imgs.shape[0],-1)

    img = (img-img.min())/(img.max()-img.min())

    print(img.min(),img.max(),imgs.min(),imgs.max())

    distList = []

    for i in range(len(imgs)):
        dist = torch.abs(imgs[i:i+1]-img).sum(dim=1)
        distList.append(dist)

    print(ind,np.array(distList).argmin(),np.array(distList).min())

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--exp_id',type=str,default="tieredimagenet")
    parser.add_argument('--viz_id',type=str,default="baseline")
    parser.add_argument('--model_id',type=str,default="dist")
    parser.add_argument('--cross_id',type=str,default="nodist_cross_loss_origfunc")
    parser.add_argument('--bcnn_id',type=str,default="nodist_bcnn")
    parser.add_argument('--nrows',type=int,default=12)
    parser.add_argument('--plot_id',type=str,default="")
    parser.add_argument('--img_inds',type=int,nargs="*")
    parser.add_argument('--classes_to_plot',type=int,nargs="*")
    #parser.add_argument('--img_to_plot',type=int,nargs="*",default=[3946,2085,2090,2119,2124,2129,2116,2121,2131])
    parser.add_argument('--img_to_plot',type=int,nargs="*",default=[2405,2410,3368,6622,6632,3238,3365,6735,6745])
    parser.add_argument('--test_on_val',action='store_true')
    parser.add_argument('--only_dist',action='store_true')

    parser.add_argument('--nb_per_class',type=int)
    args = parser.parse_args()

    vizOrder = ['gradcam', 'gradcam_pp', 'norm','rise','score_map','guided' ,'sq', 'var']
    #vizOrder = ['imgs','gradcam', 'gradcam_pp', 'score_map','guided']

    visMaps(args.exp_id,args.model_id,args.viz_id,vizOrder,args.cross_id,args.bcnn_id,args.nrows,args.img_to_plot,args.plot_id,args.test_on_val,args.only_dist)


