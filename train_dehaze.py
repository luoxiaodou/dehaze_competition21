import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

from networks.MIRNet_model import MIRNet
from losses import CharbonnierLoss
from losses import AvgMseLoss
from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

######### Model ###########
model_restoration = MIRNet()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
#print(device_ids)
#if len(device_ids)>1:
    #model_restoration = nn.DataParallel(model_restoration, device_ids = [0,1,2,3])

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-start_epoch+1, eta_min=1e-6)
else:
    warmup = True

######### Scheduler ###########
if warmup:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
#print(device_ids)
if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = [0,2,3])

######### Loss ###########
criterion = CharbonnierLoss().cuda()
criterion_mse = AvgMseLoss().cuda()
######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}

train_dataset = get_training_data(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

val_dataset = get_validation_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
#print(eval_now)
print(f'Evaluation after every {eval_now} Iterations !!!')

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0):    

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch>5:
            target, input_ = mixup.aug(target, input_)

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)  
        
        #loss = criterion(restored, target)
        loss =criterion_mse(restored,target)
    
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

