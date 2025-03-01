from typing import List, Union
import numpy as np
import torch
from torch.nn import DataParallel
import matplotlib.pyplot as plt
import json
import time
import random
import dataprepare
import dataprepare_new
import module



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_pseudocolor_image(tensor1, tensor2, output_path):
    array1 = tensor1.cpu().numpy()
    array2 = tensor2.cpu().numpy()
    difference = abs(array1 - array2)
    
    vmin = min(array1.min(), array2.min(), difference.min())
    vmax = max(array1.max(), array2.max(), difference.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    im1 = axes[0].imshow(array1, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title('Pred')
    axes[0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='horizontal')
    cbar1.ax.tick_params(labelsize=8)

    im2 = axes[1].imshow(array2, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title('Truth')
    axes[1].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='horizontal')
    cbar2.ax.tick_params(labelsize=8)

    im3 = axes[2].imshow(difference, cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title('Difference')
    axes[2].axis('off')
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='horizontal')
    cbar3.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def test(model_name, param,mu,ga, devices: Union[str, List[int]], data_num=108, batch_size=27, new_data=True):
    if isinstance(devices, List):
        device_ids = devices
    else:
        device_ids = list(range(torch.cuda.device_count()))

    model = getattr(module, model_name)(*param)
    encoder = module.Encoder()

    model.load_state_dict(torch.load(f"./model/CNN4D_best.pt"))
    encoder.load_state_dict(torch.load("./model/Encoder.pt"))

    model = DataParallel(module=model, device_ids=device_ids, output_device=device_ids[-1])
    encoder = DataParallel(module=encoder, device_ids=device_ids, output_device=0)
    
    model.to(device_ids[0])
    encoder.to(device_ids[0])

    correction = torch.nn.MSELoss()
    

    if new_data:
        with open('mu_ga_files_new.json') as f:
            data = json.load(f)
        dataset = dataprepare_new.CustomDataset(random.sample(data[str(mu)][str(ga)], data_num))
    else:
        with open('mu_ga_files.json') as f:
            data = json.load(f)
        dataset = dataprepare.CustomDataset(random.sample(data[str(mu)][str(ga)], data_num))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    start = time.time()
    total_loss = 0
    differ_loss = 0
    out_loss=0
    model.eval()
    # encoder.eval()
    
    with torch.no_grad():
        for i, item in enumerate(dataloader):
            x, Bx, y = item
            Bx = Bx.to(device_ids[0])
            x = x.to(device_ids[0])
            y = y.to(device_ids[-1])
            
            Nx = encoder(Bx)
            out = model( x,Nx)
            loss = correction(out, y)
            zero = torch.zeros_like(y)
            loss2 = correction(y, zero)
            loss3=correction(out,zero)
            differ_loss += loss2.item()
            total_loss += loss.item()
            out_loss+=loss3.item()
            
            generate_pseudocolor_image(out[0, 1, 2, :, :, 3], x[0, 1, 2, :, :, 3], f'./mid/{model_name}_{param}_re_teston_mu{mu}_ga{ga}_sample{i}.png')
        
    total_loss /= len(dataloader)
    differ_loss /= len(dataloader)
    out_loss/=len(dataloader)
    print(f'truth->0: {differ_loss}\npred->truth: {total_loss}\npred->0: {out_loss}\n TimeCost: {time.time() - start}')
    print(f'{abs(out.cpu().numpy()).mean()}')

set_random_seed(108)
# 示例调用test函数
# test('CNN4D_act', param=[3], mu=2.5,ga=0.1,devices='all', data_num=24, batch_size=8, new_data=True)
# test('CNN4D_act', param=[3], mu=1.5,ga=0.001,devices='all', data_num=24, batch_size=8, new_data=True)

mus=[1,1.5,2,2.5,3]
gas=[0.1,0.001,1e-05]

for i in range(len(mus)):
    for j in range(len(gas)):
        test('CNN4D', param=[3], mu=mus[i],ga=gas[j],devices='all', data_num=8, batch_size=8, new_data=True)
