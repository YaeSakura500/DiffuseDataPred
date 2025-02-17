import h5py
import torch
from torch.utils.data import Dataset
import os

def h5_data_read(filename, data_keys):
    file = h5py.File(filename, 'r')
    data = [torch.tensor(file[key][:].astype('float32')) for key in data_keys]
    return data

def find_velocity_filename(diffusion_filename):
    base_name = os.path.basename(diffusion_filename)
    parts = base_name.split('_')
    # Assume velocity filenames follow the pattern "gen_data_muX_Y.h5"
    velocity_filename = f"./data_diffusion/velocity/gen_data_{parts[2]}_{parts[4]}"
    return velocity_filename

class CustomDataset(Dataset):
    def __init__(self, diffusion_filenames):
        self.data = []
        for diff_file in diffusion_filenames:
            vel_file = find_velocity_filename(diff_file)
            
            if not os.path.exists(vel_file):
                print(f"Warning: Matching velocity file for {diff_file} not found.")
                continue

            f1, f2, f3 = h5_data_read(diff_file, ['f1', 'f2', 'f3'])
            u, v, w = h5_data_read(vel_file, ['u', 'v', 'w'])

            Bu = torch.zeros_like(u)
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
            Bw[:, :, -1, :] = w[:, -1, :]
            Bw[:, :, :, -1] = w[:, :, :, -1]

            u = torch.unsqueeze(u, dim=0)
            v = torch.unsqueeze(v, dim=0)
            w = torch.unsqueeze(w, dim=0)
            tar = torch.cat((u, v, w), dim=0)

            Bu = torch.unsqueeze(Bu, dim=0)
            Bv = torch.unsqueeze(Bv, dim=0)
            Bw = torch.unsqueeze(Bw, dim=0)
            bon = torch.cat((Bu, Bv, Bw), dim=0)

            f1 = torch.unsqueeze(f1, dim=0)
            f2 = torch.unsqueeze(f2, dim=0)
            f3 = torch.unsqueeze(f3, dim=0)
            f = torch.cat((f1, f2, f3), dim=0)

            self.data.append((tar, bon, f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 示例用法
filenames = ["./data_diffusion/diffusion/gen_data_mu2_ga0.1_51.h5",
            "./data_diffusion/diffusion/gen_data_mu1_ga1e-05_258.h5"]  # 替换为您的文件名列表
dataset = CustomDataset(filenames)

# 获取数据示例
for i in range(len(dataset)):
    data = dataset[i]
    print(data)
