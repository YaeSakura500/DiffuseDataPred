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


def train(model, param, epoch, devices:Union[str,List[int]],mu=1, ga=0.1, data_num=108,batch_size=27,new_data=False,RF=False):
    if isinstance(devices, List) : 
        device_ids = devices
    else: 
        device_ids = list(range(torch.cuda.device_count())) 
    
    model = getattr(module, model)(*param)
    # encoder = module.Encoder()
    if RF==True:
        model.load_state_dict(torch.load("./model/"+model.__class__.__name__+f"_{param}_best.pt"))
        # encoder.load_state_dict(torch.load("./model/Encoder.pt"))
    # model=DataParallel(module=model,device_ids=device_ids,output_device=device_ids[-1])  
    # encoder=DataParallel(module=encoder,device_ids=device_ids,output_device=0)
    model=data_parallel.BalancedDataParallel(1,module=model,device_ids=device_ids,output_device=device_ids[0])  
    # encoder=data_parallel.BalancedDataParallel(2,module=encoder,device_ids=device_ids,output_device=0)
    model=model.to(device_ids[0])

    correction = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    writer = SummaryWriter(log_dir=f'./runs/{model.module.__class__.__name__}_{param}')
    start = time.time()
    ep=0        
    best_loss = float('inf')
    best_model = None
    mu=['1','2','3']
    ga=['0.1','0.001','1e-05']
    with open('mu_ga_files.json') as f:
        data = json.load(f)

    for _ in range(30):


        datas=[]
        for j in range(len(mu)):
            for k in range(len(ga)):
                datas.extend(sample(data[mu[j]][ga[k]],13))
        dataset = dataprepare_new.CustomDataset(datas)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8,shuffle=True,drop_last=True)
        
        for _ in range(epoch):
            total_loss = 0
            model.train()

            for item in dataloader:
                optimizer.zero_grad()
                x, Bx, y = item
                x = x.to(device_ids[0])
                out = model(x)
                loss = correction(out, x)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            
            total_loss /= len(dataloader)
            writer.add_scalar('Loss/train', total_loss, ep)
            print(f'epoch: {ep}, loss: {total_loss}, TimeCost: {time.time() - start}')
            ep+=1
            
            # 判断是否包含 NaN 或 Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss contains NaN or Inf values, stopping training...")
                break
            
            # Update best model if current loss is lower
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = model.module.state_dict()
                torch.save(best_model, f"./model/{model.module.__class__.__name__}_{param}_best.pt")


    writer.close()
    torch.save(model.module.state_dict(), f"./model/{model.module.__class__.__name__}_{param}_final.pt")
    torch.save(best_model, f"./model/{model.module.__class__.__name__}_{param}_best.pt")



#使用
set_random_seed(42)
# ep=0
# for i in range(30):
train('Trans_CNN4D',param=[3,3,8,8],epoch=10,batch_size=22,devices="all",RF=False)

