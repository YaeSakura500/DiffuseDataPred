import argparse
import json
import os
from random import sample
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
from encoder_decoder import transformer
# 假设我们的数据是这个
from dataprepare_new import CustomDataset,CompressedDataSet
import module
def get_dataset():
    with open('mu_ga_files_new.json') as f:
        data = json.load(f)
    my_trainset = CustomDataset(sample(data['3']['0.1'],20))
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=1, num_workers=2, sampler=train_sampler)
    return trainloader

def get_compressed_dataset(mu,ga,batch_size):
    my_trainset = CompressedDataSet(f'mu{mu}_ga{ga}_encoded_9126.json')
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=batch_size, num_workers=2, sampler=train_sampler)
    return trainloader

def train_with_compressed(epochs,mu=1, ga=0.1,batch_size=1,RF=False):
    ### 2. 初始化我们的模型、数据、各种配置  ####
    # DDP：从外部得到local_rank参数
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')  # nccl是GPU设备上最快、最推荐的后端

    # 准备数据，要在DDP初始化之后进行
    trainloader = get_compressed_dataset(mu,ga,batch_size)

    # 构造模型
    model = transformer(d_model=9216,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=1000,batch_first=True).to(local_rank)
    param='9216_2_2_1000'
    # coder = module.Trans_CNN4D(3,3,8,6)
    # coder.load_state_dict(torch.load("./model/Trans_CNN4D_[3, 3, 8, 6, 8, 1, True]_best.pt"),strict=False)
    # encoder=coder.patch_embedding.to(local_rank)
    # DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
    ckpt_path = None
    if dist.get_rank() == 0 and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

    # DDP: 要在构造DDP model之后，才能用model初始化optimizer。
    optimizer = torch.optim.AdamW(model.parameters())

    # 假设我们的loss是这个
    loss_func = nn.MSELoss().to(local_rank)
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=f'./runs_ddp/{model.module.__class__.__name__}_'+param)
        best_loss = float('inf')
    ### 3. 网络训练  ###
    model.train()
    # encoder.eval()
    for epoch in range(epochs):
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        trainloader.sampler.set_epoch(epoch)
        total_loss=0
        # 后面这部分，则与原来完全一致了。
        for x, Bx, y in trainloader:
            optimizer.zero_grad()
            x = x.to(local_rank)
            Bx=Bx.to(local_rank)
            y=y.to(local_rank)
            # x=encoder(x)
            # Bx=encoder(Bx)
            # y=encoder(y)
            out,label = model(x,Bx,y)
            loss = loss_func(out,label)
            loss.backward()
            total_loss+=loss.item()
            optimizer.step()

        # DDP:
        # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
        #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
        # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
        if dist.get_rank() == 0:
            writer.add_scalar('Loss/train', total_loss, epoch)
                    # 判断是否包含 NaN 或 Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss contains NaN or Inf values, stopping training...")
                break
            
            # Update best model if current loss is lower
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.module, f"./model/{model.module.__class__.__name__}_"+param+"_best.pt")
    if dist.get_rank() == 0:  
        torch.save(model.module, f"./model/{model.module.__class__.__name__}_"+param+"_final.pt")
    dist.destroy_process_group()
    
def train(epochs,mu=1, ga=0.1,batch_size=1,RF=False):

    # 准备数据，要在DDP初始化之后进行
    trainloader = get_compressed_dataset(mu,ga,batch_size)

    # 构造模型
    model = transformer(d_model=9216,batch_first=True).to(local_rank)
    param='9216_6_6_1024'
    coder = module.Trans_CNN4D(3,3,8,6)
    coder.load_state_dict(torch.load("./model/Trans_CNN4D_[3, 3, 8, 6, 8, 1, True]_best.pt"),strict=False)
    encoder=coder.patch_embedding.to(local_rank)
    # DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
    ckpt_path = None
    if dist.get_rank() == 0 and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

    # DDP: 要在构造DDP model之后，才能用model初始化optimizer。
    optimizer = torch.optim.AdamW(model.parameters())

    # 假设我们的loss是这个
    loss_func = nn.MSELoss().to(local_rank)
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=f'./runs_ddp/{model.module.__class__.__name__}_'+param)
        best_loss = float('inf')
    ### 3. 网络训练  ###
    model.train()
    # encoder.eval()
    for epoch in range(epochs):
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        trainloader.sampler.set_epoch(epoch)
        total_loss=0
        # 后面这部分，则与原来完全一致了。
        for x, Bx, y in trainloader:
            optimizer.zero_grad()
            x = x.to(local_rank)
            Bx=Bx.to(local_rank)
            y=y.to(local_rank)
            x=encoder(x)
            Bx=encoder(Bx)
            y=encoder(y)
            out,label = model(x,Bx,y)
            loss = loss_func(out,label)
            loss.backward()
            total_loss+=loss.item()
            optimizer.step()

        # DDP:
        # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
        #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
        # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
        if dist.get_rank() == 0:
            writer.add_scalar('Loss/train', total_loss, epoch)
                    # 判断是否包含 NaN 或 Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss contains NaN or Inf values, stopping training...")
                break
            
            # Update best model if current loss is lower
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.module, f"./model/{model.module.__class__.__name__}_"+param+"_best.pt")
    if dist.get_rank() == 0:  
        torch.save(model.module, f"./model/{model.module.__class__.__name__}_"+param+"_final.pt")
    dist.destroy_process_group()

train_with_compressed(epochs=20)