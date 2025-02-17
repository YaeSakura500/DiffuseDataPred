from typing import List, Union
import torch
import dataprepare
import dataprepare_new
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import device
from torch.nn.modules.module import T
from Conv4d import Conv4d
import module
import time
import json
import random
from random import sample
from torch.nn.parallel import DataParallel
import numpy as np
import transformer
import data_parallel

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, param, mu, ga, epoch, devices:Union[str,List[int]], data_num=108,batch_size=27,new_data=False,RF=False):
    if isinstance(devices, List) : 
        device_ids = devices
    else: 
        device_ids = list(range(torch.cuda.device_count())) 
    
    model = getattr(module, model)(*param)
    encoder = module.Encoder()
    if RF==True:
        model.load_state_dict(torch.load("./model/"+model.__class__.__name__+"_best.pt"))
        encoder.load_state_dict(torch.load("./model/Encoder.pt"))
    # model=DataParallel(module=model,device_ids=device_ids,output_device=device_ids[-1])  
    # encoder=DataParallel(module=encoder,device_ids=device_ids,output_device=0)
    model=data_parallel.BalancedDataParallel(2,module=model,device_ids=device_ids,output_device=device_ids[-1])  
    encoder=data_parallel.BalancedDataParallel(2,module=encoder,device_ids=device_ids,output_device=0)
    model.to(device_ids[0])
    encoder.to(device_ids[0])

    correction = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    
    with open('mu_ga_files.json') as f:
        data = json.load(f)
    if new_data==True:
        dataset = dataprepare_new.CustomDataset(sample(data[str(mu)][str(ga)], data_num))
    else:
        dataset = dataprepare.CustomDataset(sample(data[str(mu)][str(ga)], data_num))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    writer = SummaryWriter(log_dir=f'./runs/{model.module.__class__.__name__}_{param}_ga{ga}_mu{mu}')
    start = time.time()
    
    best_loss = float('inf')
    best_model = None

    for i in range(epoch):
        total_loss = 0
        model.train()
        encoder.eval()

        for item in dataloader:
            optimizer.zero_grad()
            x, Bx, y = item
            Bx = Bx.to(device_ids[0])
            x = x.to(device_ids[0])
            y = y.to(device_ids[-1])
            
            Nx = encoder(Bx)
            out = model(x, Nx)
            loss = correction(out, y)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        total_loss /= len(dataloader)
        writer.add_scalar('Loss/train', total_loss, i)
        print(f'epoch: {i}, loss: {total_loss}, TimeCost: {time.time() - start}')
        
        # 判断是否包含 NaN 或 Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Loss contains NaN or Inf values, stopping training...")
            break
        
        # Update best model if current loss is lower
        if total_loss < best_loss:
            best_loss = total_loss
            best_model = model.module.state_dict()

    writer.close()
    torch.save(model.module.state_dict(), f"./model/{model.module.__class__.__name__}_{param}_final.pt")
    torch.save(best_model, f"./model/{model.module.__class__.__name__}_{param}_best.pt")
    # torch.save(encoder.state_dict(), f"./model/{model.module.__class__.__name__}_encoder_final.pt")



#使用
set_random_seed(42)
train('Trans_CNN4D',param=[3,3,8,8,8,2],mu=3,ga=0.001,epoch=200,data_num=24,batch_size=22,devices="all",RF=False)
train('Trans_gong',param=[3,3,8,8,8,2],mu=1,ga=0.1,epoch=200,batch_size=16,devices="all",RF=True)