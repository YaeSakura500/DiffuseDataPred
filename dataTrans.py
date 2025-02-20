import os
import json
from random import sample
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataprepare_new import CustomDataset 

import module 

def get_dataset(mu,ga):
    with open('mu_ga_files_new.json') as f:
        data = json.load(f)
    my_trainset = CustomDataset(sample(data[f'{mu}'][f'{ga}'],120))
    trainloader = DataLoader(my_trainset, batch_size=1, num_workers=2)
    return trainloader

def save_transformed_data(data_loader, encoder, save_path):
    encoder.eval()  # 设置encoder为评估模式
    transformed_data = []

    with torch.no_grad():
        for x, Bx, y in data_loader:
            x = x.to(0)
            Bx = Bx.to(0)
            y = y.to(0)
            x = encoder(x)
            Bx = encoder(Bx)
            y = encoder(y)
            transformed_data.append((x.cpu().numpy().tolist(), Bx.cpu().numpy().tolist(), y.cpu().numpy().tolist()))

    with open(save_path, 'w') as f:
        json.dump(transformed_data, f)

def main(mu,ga):
    trainloader = get_dataset(mu,ga)


    coder = module.Trans_CNN4D(3, 3, 8, 6)
    coder.load_state_dict(torch.load("./model/Trans_CNN4D_[3, 3, 8, 6, 8, 1, True]_best.pt"), strict=False)
    encoder = coder.patch_embedding.to(0)

    save_transformed_data(trainloader, encoder, f'mu{mu}_ga{ga}_encoded_9126.json')

mus=[1,2,3]
gas=[0.1,0.001,1e-05]
for mu in mus:
    for ga in gas:
        main(mu,ga)
