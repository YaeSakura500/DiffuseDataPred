import json
import time
import torch
import random
import numpy as np
from thop import profile
from thop import clever_format
import matplotlib.pyplot as plt

import dataprepare_new


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#使用
set_random_seed(42)


def generate_pseudocolor_image(tensor1, tensor2, output_path):
    # 将tensor转换为numpy数组
    array1 = tensor1.cpu().numpy()
    array2 = tensor2.cpu().numpy()
    
    # 计算差值
    difference = abs(array1 - array2)
    
    # 找到所有数组中的最小值和最大值
    vmin = min(array1.min(), array2.min(), difference.min())
    vmax = max(array1.max(), array2.max(), difference.max())
    
    # 创建伪色图
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


# 示例用法
# tensor1 = torch.randn(100, 100)
# tensor2 = torch.randn(100, 100)
# generate_pseudocolor_image(tensor1, tensor2, 'output_image.png')




device_ids = list(range(torch.cuda.device_count())) 
# model = module.Trans_new(3,3,16,16,8,8)
# model = torch.load('models/Trans_gong_model_8.pt')
# encoder=torch.load('models/Trans_gong_Encoder.pt')
# model = torch.load('models/CNN_model_mu1_ga0.1.pt')
# encoder=torch.load('models/CNN_Encoder.pt')
model = torch.load('models/CNN_model_mu1_ga0.1.pt')
encoder=torch.load('models/CNN_Encoder.pt')
# model = torch.load('models/divCNN_model_3.pt')
# encoder=torch.load('models/divCNN_Encoder.pt')
# model = torch.load('models/TriditionCNN_model_3.pt')
# encoder=torch.load('models/TriditionCNN_Encoder.pt')
# model=DataParallel(module=model,device_ids=device_ids,output_device=7)
# encoder=DataParallel(module=encoder,device_ids=device_ids,output_device=7)
# model=model.module
# encoder=encoder.module
model.to(device_ids[0])
encoder.to(device_ids[0])
correction=torch.nn.MSELoss()
with open('mu_ga_files_new.json') as f:
    data=json.load(f)
print('loading data')
dataset=dataprepare_new.CustomDataset(random.sample(data['3']['1e-05'],12))
print('prepare dataloader')
dataloader = torch.utils.data.DataLoader(dataset,4,num_workers=0)
print('data loaded')


start=time.time()
total_loss=0
differ_loss=0
out_loss=0
model.eval()
encoder.eval()
with torch.no_grad():
    i=0
    for item in dataloader:
        x,Bx,y=item
        Bx=Bx.to(device_ids[0])
        x = x.to(device_ids[0])
        y=y.to(device_ids[7])
        Nx=encoder(Bx)
        out = model(x,Bx)
        loss=correction(out,y)
        zero=torch.zeros_like(y)
        loss2=correction(y,zero)
        loss3=correction(out,zero)
        differ_loss+=loss2.item()
        total_loss+=loss.item()
        out_loss==loss3.item()
        generate_pseudocolor_image(out[0,1,2,:,:,3],y[0,1,2,:,:,3],f'./mid/CNN_model_mu1_ga0.1_teston_mu3_ga1e-05{i}.png')
        i+=1
    total_loss/=len(dataloader)
    differ_loss/=len(dataloader)
    print(f'truth->0: {differ_loss}\npred->truth: {total_loss}\npred->0: {out_loss}\n TimeCost: {time.time() - start}')
print(f'{abs(out.cpu().numpy()).mean()}')
torch.save(model.module.state_dict(),'./mid/models/CNN4d_3_best.pt')
torch.save(encoder.module.state_dict(),'./mid/models/CNN4d_encoder_best.pt')

# input_ex=(torch.randn(1,3,64,64,64,64,device='cuda:0'),torch.randn(1,3,64,64,64,64,device='cuda:0'))
# flops,params=profile(model,input_ex)
# flops, params = clever_format([flops, params], '%.3f')
# print(f"运算量：{flops}, 参数量：{params}")




