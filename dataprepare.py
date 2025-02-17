import h5py
import torch
from torch.utils.data import Dataset

def h5_data_read(filename):
    file = h5py.File(filename, 'r')
    u = torch.tensor(file['u'][:].astype('float32'))
    v = torch.tensor(file['v'][:].astype('float32'))
    w = torch.tensor(file['w'][:].astype('float32'))
    f1 = torch.tensor(file['f1'][:].astype('float32'))
    f2 = torch.tensor(file['f2'][:].astype('float32'))
    f3 = torch.tensor(file['f3'][:].astype('float32'))
    return u, v, w, f1, f2, f3

class CustomDataset(Dataset):
    def __init__(self, filenames):
        self.data = []
        for filename in filenames:
            u, v, w, f1, f2, f3 = h5_data_read(filename)
            Bu=torch.zeros_like(u)
            Bv = torch.zeros_like(v)
            Bw = torch.zeros_like(w)
            Bu[0,:,:,:]=u[0,:,:,:]
            Bu[:, 0, :, :] = u[:, 0, :, :]
            Bu[:, :, 0, :] = u[:, :, 0, :]
            Bu[:, :, :, 0] = u[:, :, :, 0]
            Bu[-1,:,:,:]=u[-1,:,:,:]
            Bu[:, -1, :, :] = u[:, -1, :, :]
            Bu[:, :, -1, :] = u[:, :, -1, :]
            Bu[:, :, :, -1] = u[:, :, :, -1]
            Bv[0,:,:,:]=v[0,:,:,:]
            Bv[:, 0, :, :] = v[:, 0, :, :]
            Bv[:, :, 0, :] = v[:, :, 0, :]
            Bv[:, :, :, 0] = v[:, :, :, 0]
            Bv[-1,:,:,:]=v[-1,:,:,:]
            Bv[:, -1, :, :] = v[:, -1, :, :]
            Bv[:, :, -1, :] = v[:, :, -1, :]
            Bv[:, :, :, -1] = v[:, :, :, -1]
            Bw[0,:,:,:]=w[0,:,:,:]
            Bw[:, 0, :, :] = w[:, 0, :, :]
            Bw[:, :, 0, :] = w[:, :, 0, :]
            Bw[:, :, :, 0] = w[:, :, :, 0]
            Bw[-1,:,:,:]=w[-1,:,:,:]
            Bw[:, -1, :, :] = w[:, -1, :, :]
            Bw[:, :, -1, :] = w[:, :, -1, :]
            Bw[:, :, :, -1] = w[:, :, :, -1]
            u= torch.unsqueeze(u,dim=0)
            v = torch.unsqueeze(v, dim=0)
            w = torch.unsqueeze(w, dim=0)
            tar=torch.cat((u,v,w),dim=0)
            Bu= torch.unsqueeze(Bu,dim=0)
            Bv = torch.unsqueeze(Bv, dim=0)
            Bw = torch.unsqueeze(Bw, dim=0)
            bon=torch.cat((Bu,Bv,Bw),dim=0)
            f1 = torch.unsqueeze(f1, dim=0)
            f2 = torch.unsqueeze(f2, dim=0)
            f3 = torch.unsqueeze(f3, dim=0)
            f=torch.cat((f1,f2,f3),dim=0)
            self.data.append((tar,bon,f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 示例用法
# filenames = ["./data_diffusion/diffusion/gen_data_mu2_ga0.1_51.h5",
#             "./data_diffusion/diffusion/gen_data_mu1_ga1e-5_258.h5"]  # 替换为您的文件名列表
# dataset = CustomDataset(filenames)

# # 获取数据示例
# for i in range(len(dataset)):
#     data = dataset[i]
#     print(data)
