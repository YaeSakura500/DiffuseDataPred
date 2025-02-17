import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):   # input (b,c,h,w,t)
    """
    对3D图像作Patch Embedding操作
    """
    def __init__(self, img_size=32, patch_size=8, in_c=4, embed_dim=256, norm_layer=None):
        """
        此函数用于初始化相关参数
        :param img_size: 输入图像的大小
        :param patch_size: 一个patch的大小
        :param in_c: 输入图像的通道数
        :param embed_dim: 输出的每个token的维度
        :param norm_layer: 指定归一化方式，默认为None
        """
        super(PatchEmbed3D, self).__init__()
        img_size = (img_size, img_size, img_size)  # 224 -> (224, 224)
        patch_size = (patch_size, patch_size, patch_size)  # 16 -> (16, 16)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])  # 计算原始图像被划分为(14, 14)个小块
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]  # 计算patch的个数为14*14=196个
        # 定义卷积层
        self.proj = nn.Conv3d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.relu = nn.ReLU()  # 非线性
        # 定义归一化方式
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        此函数用于前向传播
        :param x: 原始图像
        :return: 处理后的图像
        """
        # 对图像依次作卷积、展平和调换处理: [B, C, H, W, T] -> [B, C, HWT] -> [B, T, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.relu(x)  # 非线性
        # 归一化处理
        x = self.norm(x)
        return x
    

class PatchDecoder(nn.Module):   # input (b,c,h,w,t)
    """
    对3D图像作Patch decoder操作
    """
    def __init__(self, img_size=32, patch_size=8, in_c=32, out_dim=3, norm_layer=None):
        """
        此函数用于初始化相关参数
        :param img_size: 输入图像的大小
        :param patch_size: 一个patch的大小
        :param in_c: 输入图像的通道数
        :param embed_dim: 输出的每个token的维度
        :param norm_layer: 指定归一化方式，默认为None
        """
        super(PatchDecoder, self).__init__()
        self.patch_num = int(img_size / patch_size)
        img_size = (img_size, img_size, img_size)  # 224 -> (224, 224)
        patch_size = (patch_size, patch_size, patch_size)  # 16 -> (16, 16)
        self.img_size = img_size
        self.inchannel=in_c
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])  # 计算原始图像被划分为(14, 14)个小块
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]  # 计算patch的个数为14*14=196个
        # 定义卷积层
        # self.proj = nn.Conv3d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size).to(DEVICE)
        self.proj = nn.ConvTranspose3d(in_channels=in_c, out_channels=out_dim, kernel_size=patch_size, stride=patch_size)
        self.relu = nn.ReLU()
        # 定义归一化方式
        self.norm = norm_layer(out_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        此函数用于前向传播
        :param x: 原始图像
        :return: 处理后的图像
        """
        x = x.reshape([-1,self.inchannel , self.patch_num, self.patch_num, self.patch_num])
        # 对图像依次作卷积、展平和调换处理: [B, C, H, W, T] -> [B, C, HWT] -> [B, HWT, C]
        x = self.proj(x)
        # x = self.relu(x)  # 加非线性
        # 归一化处理
        x = self.norm(x)
        return x
    
# test=torch.randn((7,4,32,32,32))
# model=PatchEmbed3D()
# out=model(test)