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
import data_parallel

from torch.utils.tensorboard import SummaryWriter

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
from encoder_decoder import transformer
# 假设我们的数据是这个
from dataprepare_new import CustomDataset,CompressedDataSet
import module
from loss import Not0Loss
def get_dataset(mu,ga,batch_size):
    with open('mu_ga_files_new.json') as f:
        data = json.load(f)
    my_trainset = CustomDataset(sample(data[f'{mu}'][f'{ga}'],30))
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=batch_size, num_workers=2,shuffle=True)
    return trainloader


def get_compressed_dataset(mu,ga,batch_size):
    my_trainset = CompressedDataSet(f'mu{mu}_ga{ga}_encoded_9126.json')
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=batch_size, num_workers=2,drop_last=True,shuffle=True)
    return trainloader

def train_with_compressed(epochs,mu=1, ga=1e-05,batch_size=20,RF=False):
    device_ids = list(range(torch.cuda.device_count())) 
    trainloader = get_compressed_dataset(mu,ga,batch_size)

    model = transformer(d_model=9216,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=2896,batch_first=True).to(device_ids[0])
    param='9216_2_2_2896'

    # DP: 构造DP model
    model = data_parallel.BalancedDataParallel(0,module=model, device_ids=device_ids, output_device=device_ids[-1])


    optimizer = torch.optim.AdamW(model.parameters())

    # 假设我们的loss是这个
    loss_func = Not0Loss
    writer = SummaryWriter(log_dir=f'./runs_dp/{model.module.__class__.__name__}_'+param)
    best_loss = float('inf')
    ### 3. 网络训练  ###
    model.train()
    for epoch in range(epochs):
        total_loss=0
        if epoch==9:
            model.module.set_simple_train(False)
        for x, Bx, y in trainloader:
            optimizer.zero_grad()
            x = x.to(device_ids[0])
            Bx=Bx.to(device_ids[0])
            y=y.to(device_ids[0])
            out,label = model(x,Bx,y)
            loss = loss_func(out,label,ep=0)
            loss.backward()
            total_loss+=loss.item()
            optimizer.step()


        writer.add_scalar('Loss/train', total_loss, epoch)
                # 判断是否包含 NaN 或 Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Loss contains NaN or Inf values, stopping training...")
            break
            
        # Update best model if current loss is lower
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.module.state_dict(), f"./model/{model.module.__class__.__name__}_"+param+"_best.pt")
 
    torch.save(model.module.state_dict(), f"./model/{model.module.__class__.__name__}_"+param+"_final.pt")

    
def train(epochs,mu=1, ga=1e-05,batch_size=5,RF=False):
    device_ids=[1,2,3,4,5,6]
    # 准备数据，要在DDP初始化之后进行
    trainloader = get_dataset(mu,ga,batch_size)

    # 构造模型
    model = transformer(d_model=9216,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=2896,batch_first=True).to(device_ids[0])
    model.load_state_dict(torch.load(f"./model/transformer_9216_2_2_2896_best.pt"))
    param='9216_2_2_2896'
    coder = module.Trans_CNN4D(3,3,8,6)
    coder.load_state_dict(torch.load("./model/Trans_CNN4D_[3, 3, 8, 6, 8, 1, True]_best.pt"),strict=False)
    encoder=coder.patch_embedding.to(0)
    decoder=coder.patch_decoding.to(7)
    # DDP: 构造DDP model
    model=data_parallel.BalancedDataParallel(0,module=model, device_ids=device_ids, output_device=7)

    # DDP: 要在构造DDP model之后，才能用model初始化optimizer。
    optimizer = torch.optim.AdamW([
        {'params':model.parameters()},
        {'params':encoder.parameters()},
        {'params':decoder.parameters()}
        ])

    # 假设我们的loss是这个
    loss_func = Not0Loss
    writer = SummaryWriter(log_dir=f'./runs_ddp/{model.module.__class__.__name__}_'+param)
    best_loss = float('inf')
    ### 3. 网络训练  ###
    model.train()
    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        total_loss=0
        # 后面这部分，则与原来完全一致了。
        for x, Bx, Y in trainloader:
            optimizer.zero_grad()
            x = x.to(0)
            Bx=Bx.to(0)
            y=Y.to(0)
            x=encoder(x).to(1)
            Bx=encoder(Bx).to(1)
            y=encoder(y).to(1)
            out,_ = model(x,Bx,y)
            out=decoder(out)
            label=Y.to(7)
            loss = loss_func(out,label)
            loss.backward()
            total_loss+=loss.item()
            optimizer.step()

        writer.add_scalar('Loss/train', total_loss/len(trainloader), epoch)
                # 判断是否包含 NaN 或 Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Loss contains NaN or Inf values, stopping training...")
            break
            
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.module.state_dict(), f"./model/{model.module.__class__.__name__}_"+param+"_best.pt")
 
    torch.save(model.module.state_dict(), f"./model/{model.module.__class__.__name__}_"+param+"_final.pt")

# train_with_compressed(epochs=80)
train(epochs=10)