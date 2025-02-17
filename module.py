from typing import Optional
import torch
from Conv4d import Conv4d
from Conv4dTranspose import ConvTranspose4d
import torch.nn as nn
import time
import logging
import transformer
import enc
from thop import profile
from thop import clever_format


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.convs= nn.ModuleList( [Conv4d(3,3,
                          kernel_size=3,
                          padding=1,
                          stride=1
                          )for _ in range(12)])

    def forward(self,x):
        out=x
        for layer in self.convs:
            out=out+layer(out)
        out=out+x
        return out


class CNN4D(nn.Module):
    def __init__(self,layers):
        super(CNN4D,self).__init__()
        self.layers=layers
        self.gathers=nn.ModuleList([Conv4d(6,3,kernel_size=3,padding=1)for _ in range(layers)])
        self.convs1=nn.ModuleList([Conv4d(3,3,kernel_size=(3,3,3,1),padding=(1,1,1,0))for _ in range(layers)])
        self.convs2=nn.ModuleList([Conv4d(3,3,kernel_size=(1,1,1,3),padding=(0,0,0,1))for _ in range(layers)])
        self.decents = nn.ModuleList([Conv4d(3, 3, kernel_size=1, padding=0) for _ in range(layers)])

    def forward(self,f,bondary):
        out=f
        for i in range(self.layers):
            temp=torch.cat((out,bondary),dim=1)
            temp=self.gathers[i](temp)
            temp=self.convs1[i](temp)
            out=self.convs2[i](temp)+self.decents[i](out)
        return out
    
class CNN4D_act(nn.Module):
    def __init__(self,layers):
        super(CNN4D_act,self).__init__()
        self.layers=layers
        self.gathers=nn.ModuleList([Conv4d(6,3,kernel_size=3,padding=1)for _ in range(layers)])
        self.convs1=nn.ModuleList([Conv4d(3,3,kernel_size=(3,3,3,1),padding=(1,1,1,0))for _ in range(layers)])
        self.convs2=nn.ModuleList([Conv4d(3,3,kernel_size=(1,1,1,3),padding=(0,0,0,1))for _ in range(layers)])
        self.decents = nn.ModuleList([Conv4d(3, 3, kernel_size=1, padding=0) for _ in range(layers)])
        self.act=nn.Tanh()

    def forward(self,f,bondary):
        out=f
        for i in range(self.layers):
            if i !=(self.layers-1):
                out=self.act(out)
            temp=torch.cat((out,bondary),dim=1)
            temp=self.gathers[i](temp)
            temp=self.convs1[i](temp)
            out=self.convs2[i](temp)+self.decents[i](out)

        return out
    
# class Triditional_CNN4D(nn.Module):
#     def __init__(self,layers):
#         super(Triditional_CNN4D,self).__init__()
#         self.layers=layers
#         self.gathers=nn.ModuleList([Conv4d(6,3,kernel_size=3,padding=1)for _ in range(layers)])
#         self.convs=nn.ModuleList([Conv4d(3,3,kernel_size=(3,3,3,3),padding=(1,1,1,1))for _ in range(layers)])
#         self.decents = nn.ModuleList([Conv4d(3, 3, kernel_size=1, padding=0) for _ in range(layers)])

#     def forward(self,f,bondary):
#         out=f
#         for i in range(self.layers):
#             temp=torch.cat((out,bondary),dim=1)
#             temp=self.gathers[i](temp)
#             out=self.convs[i](temp)+self.decents[i](out)
#         return out

# class div_CNN4D(nn.Module):
#     def __init__(self,layers):
#         super(div_CNN4D,self).__init__()
#         self.layers=layers
#         self.gathers=nn.ModuleList([Conv4d(6,3,kernel_size=3,padding=1)for _ in range(layers)])
#         self.convs1=nn.ModuleList([Conv4d(3,3,kernel_size=(3,1,1,1),padding=(1,0,0,0))for _ in range(layers)])
#         self.convs2=nn.ModuleList([Conv4d(3,3,kernel_size=(1,3,1,1),padding=(0,1,0,0))for _ in range(layers)])
#         self.convs3=nn.ModuleList([Conv4d(3,3,kernel_size=(1,1,3,1),padding=(0,0,1,0))for _ in range(layers)])
#         self.convs4=nn.ModuleList([Conv4d(3,3,kernel_size=(1,1,1,3),padding=(0,0,0,1))for _ in range(layers)])
#         self.decents = nn.ModuleList([Conv4d(3, 3, kernel_size=1, padding=0) for _ in range(layers)])

#     def forward(self,f,bondary):
#         out=f
#         for i in range(self.layers):
#             temp=torch.cat((out,bondary),dim=1)
#             temp=self.gathers[i](temp)
#             temp=self.convs1[i](temp)
#             temp=self.convs2[i](temp)
#             temp=self.convs3[i](temp)
#             out=self.convs4[i](temp)+self.decents[i](out)
#         return out

# class Trans(nn.Module):
#     def __init__(self,layers):
#         super(Trans,self).__init__()
#         self.layers=layers
#         self.encode=nn.Linear(64**3,)
#         self.MLPs=nn.ModuleList(
#             [
#                 nn.Linear(8**3,8**3) for _ in range(self.layers)
#             ]
#         )
#         self.Trans=nn.ModuleList(
#             [
#                 nn.TransformerDecoderLayer(8**3,8**3)
#             ]
#         )
#
#     def forward(self,f,bondary):
#         b,w,h,l,t=f.shape
#         f=torch.einsum('BWHLT->BTWHL',f)
#         bondary = torch.einsum('BWHLT->BTWHL', bondary)
#         input=torch.cat((f,bondary),dim=1)
#         # input=self.encode(input)
#         for i in self.encode:
#             input=i(input)
#         input=input.reshape(b,t,-1)
#         input=self.MLP1(input)
#         out=self.layer(input,input)
#         out=self.MLP2(out)
#         out=out.reshape(b,t,16,16,16)
#         out = self.decode(out)
#         out=torch.einsum('BTWHL->BWHLT',out)
#         return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Sequential(
            nn.Linear(in_channels * patch_size ** 3,2*in_channels * patch_size ** 3),
            nn.Linear(2*in_channels * patch_size ** 3, embedding_dim))

    def forward(self, x):
        B, C, X, Y, Z, T = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, T, -1, self.patch_size, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5, 6).contiguous().view(B,-1, patches.size(3), self.patch_size ** 3 * C)
        embedded_patches = self.embedding(patches)
        return embedded_patches.view(B, T, -1)


class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers=num_layers
        self.mth_atten=nn.ModuleList(
            [
                nn.TransformerDecoderLayer(embedding_dim,num_heads,dim_feedforward=dim_feedforward,batch_first=True) for _ in range(2*num_layers)
            ]
        )
        self.crossatten=nn.ModuleList(
            [
                nn.MultiheadAttention(embedding_dim,num_heads,batch_first=True) for _ in range(num_layers)
            ]
        )
        self.MLPs=nn.ModuleList(
            [
                nn.Linear(embedding_dim,embedding_dim)  for _ in range(num_layers)
            ]
        )

    def forward(self, x,Bond):
        out=x
        for i in range(self.layers):
            out=self.mth_atten[i](x,x)
            out=self.crossatten[i](Bond,out,out,need_weights=False)[0]
            out=self.mth_atten[i+self.layers](out,out)+x
            x=self.MLPs[i](out)
        return x

class SimpleTransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dim_feedforward=2048):
        super(SimpleTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers=num_layers
        self.mth_atten=nn.ModuleList(
            [
                nn.TransformerDecoderLayer(embedding_dim,num_heads,dim_feedforward=dim_feedforward,batch_first=True) for _ in range(2*num_layers)
            ]
        )
        self.crossatten=nn.ModuleList(
            [
                nn.MultiheadAttention(embedding_dim,num_heads,batch_first=True) for _ in range(num_layers)
            ]
        )
        # self.MLP=nn.Sequential(
        #         nn.Linear(embedding_dim,embedding_dim) ,
        #         nn.Linear(embedding_dim,embedding_dim)
        #     )

    def forward(self, x,Bond):
        out=x
        for i in range(self.layers):
            out=self.mth_atten[i](x,x)
            out=self.crossatten[i](Bond,out,out,need_weights=False)[0]
            out=self.mth_atten[i+self.layers](out,out)+x
        # x=self.MLP(out)
        return x

class PatchDecoding(nn.Module):
    def __init__(self, embedding_dim, out_channels, patch_size):
        super(PatchDecoding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim=embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2*out_channels * patch_size ** 3),
            nn.SELU(),
            nn.Linear(2*out_channels * patch_size ** 3,out_channels * patch_size ** 3))

    def forward(self, x, original_shape):
        B, T, _ = x.shape
        C, X, Y, Z = original_shape
        x=x.view(B,T,-1,self.embedding_dim)
        decoded_patches = self.decoder(x)
        decoded_patches = decoded_patches.view(B, T, -1, C, self.patch_size, self.patch_size, self.patch_size)
        patches = decoded_patches.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
        patches = patches.view(B, C, X, Y, Z,T)
        return patches

class PatchEncodingCNN4D(nn.Module):
    def __init__(self,in_channel,input_size,patch_size,embedding_dim):
        super(PatchEncodingCNN4D,self).__init__()
        self.input_size=input_size
        self.patch_size=patch_size
        self.embedding_dim=embedding_dim
        self.in_channel=in_channel
        self.patch_enc=Conv4d(
            in_channels=in_channel,
            out_channels=in_channel*embedding_dim,
            kernel_size=(patch_size,patch_size,patch_size,1),
            stride=(patch_size,patch_size,patch_size,1))
        # self.enshape=Conv4d(
        #     in_channels=in_channel*embedding_dim,
        #     out_channels=in_channel*embedding_dim*((input_size//patch_size)**3),
        #     kernel_size=((input_size//patch_size),(input_size//patch_size),(input_size//patch_size),1),
        #     stride=(1,1,1,1)
        # )
    
    def forward(self,x):
        out=self.patch_enc(x)
        # out=self.enshape(out)
        out=out.permute(0,5,1,2,3,4)
        # out=out.permute(0,5,1,2,3,4).flatten(2)
        return out
    
class PatchDecodingCNN4D(nn.Module):
    def __init__(self,out_channel,input_size,patch_size,embedding_dim):
        super(PatchDecodingCNN4D,self).__init__()
        self.input_size=input_size
        self.patch_size=patch_size
        self.embedding_dim=embedding_dim
        self.in_channel=out_channel
        self.patch_dedec1=ConvTranspose4d(
            out_channels=out_channel*2,
            in_channels=out_channel*embedding_dim,
            kernel_size=(patch_size//2,patch_size//2,patch_size//2,1),
            stride=(patch_size//2,patch_size//2,patch_size//2,1))
        self.patch_dedec2=ConvTranspose4d(
            out_channels=out_channel*2,
            in_channels=out_channel*2,
            kernel_size=(4,4,4,1),
            stride=(4,4,4,1))
        self.patch_dedec3=Conv4d(
            out_channels=out_channel,
            in_channels=out_channel*2,
            kernel_size=(4,4,4,1),
            stride=(2,2,2,1),
            padding=(1,1,1,0))
        self.act=nn.Tanh()
    
    def forward(self,x):
        # b,t,_=x.shape
        # out=x.reshape(b,t,-1,self.input_size//self.patch_size,self.input_size//self.patch_size,self.input_size//self.patch_size).permute(0,2,3,4,5,1)
        out=x.permute(0,2,3,4,5,1)
        # out=self.enshape(out)
        out=self.patch_dedec1(out)
        out=self.act(out)
        out=self.patch_dedec2(out)
        out=self.act(out)
        out=self.patch_dedec3(out)
        return out

class Trans_new(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, embedding_dim, num_heads, num_layers):
        super(Trans_new, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.transformer = TransformerModel(int(embedding_dim*((64/patch_size)**3)), num_heads, num_layers)
        self.patch_decoding = PatchDecoding(embedding_dim, out_channels, patch_size)

    def forward(self, x,Bond):
        original_shape = x.shape[1:5]
        x = self.patch_embedding(x)
        Bond=self.patch_embedding(Bond)
        x = self.transformer(x,Bond)
        x = self.patch_decoding(x, original_shape)
        return x

    def train_enc(self,x):
        original_shape=x.shape[1:5]
        x=self.patch_embedding(x)
        y=self.patch_decoding(x,original_shape)
        return y


class Trans_CNN4D(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, embedding_dim):
        super(Trans_CNN4D, self).__init__()
        self.patch_embedding = PatchEncodingCNN4D(in_channels,64, patch_size, embedding_dim)
        self.patch_decoding = PatchDecodingCNN4D(out_channels,64, patch_size,embedding_dim)
    def forward(self, x):
        x=self.emb(x)
        out=self.dec(x)  
        return out
    
    def emb(self,x):
        return self.patch_embedding(x)
    
    def dec(self,x):
        return self.patch_decoding(x)
    
    
    
class Trans(nn.Module):
    def __init__(self,embdim,num_heads,numlayers):
        super(Trans, self).__init__()
        self.trans=nn.Transformer(embdim,num_heads,numlayers,numlayers,batch_first=True)
    def forward(self, x,b):
        out=self.trans(x,b)
        return out


class Trans_gen(nn.Module):
    def __init__(self,embdim,num_heads,numlayers,simple_train:bool = False,T: Optional[int] = None):
        super(Trans_gen, self).__init__()
        self.gather=nn.Linear(2*embdim,embdim)
        self.trans=nn.Transformer(embdim,num_heads,numlayers,numlayers,batch_first=True)
        self.ffd=nn.Linear(embdim,embdim)
        self.simple_train=simple_train # 使用真值作为输入，减少累计误差
        self.T=T # 预测步数
    
    def forward(self, x,b,y):
        shape=x.shape
        x=x.flatten(2)
        b=b.flatten(2)
        y=y.flatten(2)
        input=self.gather(torch.cat((x,b),dim=-1))
        if self.T is None:
            T=input.shape[1]
        else:
            T=self.T
            if T>y.shape[1]:
                T=y.shape[1]
        if self.simple_train==True:
            for i in range(T):
                out=self.trans(input,input)
                y=torch.cat((y,out),dim=1)
                input=torch.cat((input,y[::,T:T+1:,]),dim=1)
            out=self.ffd(y[::,-T::,])
        else:
            for i in range(T):
                out=self.trans(input,input)[::,-1::,]
                out=self.ffd(out)
                input=torch.cat((input,out),dim=1)
            out=input[::,-T::,]
        out=out.reshape(shape[0],T,shape[2],shape[3],shape[4],shape[5])
        y=y[::,0:T:,].reshape(shape[0],T,shape[2],shape[3],shape[4],shape[5])
        
        return out,y
    
    def gen(self,x,b):
        shape=x.shape
        x=x.flatten(2)
        b=b.flatten(2)
        input=self.gather(torch.cat((x,b),dim=-1))
        T=input.shape[1]
        for i in range(T):
            out=self.trans(input,input)[::,-1::,]
            out=self.ffd(out)
            input=torch.cat((input,out),dim=1)
        out=input[::,-T::,]
        out=out.reshape(shape)
        return out
    
    def setT(self,T: Optional[int]):
        self.T=T
    
    def set_simple_train(self,simple_train:bool):
        self.simple_train=simple_train



class Trans_gong(nn.Module):
    def __init__(self,d_model,ff_d, num_heads, num_layers):
        super(Trans_gong, self).__init__()
        self.transformer = transformer.Transformer(N=num_layers,d_model=d_model,ff_d=ff_d,head_num=num_heads, dropout=0)

    def forward(self, x,Bond):
        batch=transformer.data_gen(Bond,x)
        y = self.transformer(batch.src1,batch.trg1,batch.src_mask,batch.trg_mask)
        y=torch.cat((Bond[:,0:1,:,],y),dim=1)
        return y
    


class Trans_gong_all(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, embedding_dim, num_heads, num_layers):
        super(Trans_gong_all, self).__init__()
        self.patch_embedding = enc.PatchEmbed3D(img_size=64, patch_size=patch_size,in_c=in_channels,embed_dim=embedding_dim)
        self.transformer = transformer.Transformer(N=num_layers,d_model=int(embedding_dim*((64/patch_size)**3)),ff_d=2048,head_num=num_heads, dropout=0)
        self.patch_decoding = enc.PatchDecoder(img_size=64, patch_size=patch_size,in_c=embedding_dim,out_dim=out_channels)

    def forward(self, x,Bond):
        original_shape = x.shape
        times=x.shape[-1]
        for i in range(times):
            inx=x[:,:,:,:,:,i].squeeze(-1)
            inx = self.patch_embedding(inx).unsqueeze(1)
            b=Bond[:,:,:,:,:,i].squeeze(-1)
            b=self.patch_embedding(b).unsqueeze(1)
            if i==0:
                B=b
                Inx=inx
                continue
            Inx=torch.concat((Inx,inx),dim=1)
            B=torch.concat((B,b),dim=1)
        Inx=Inx.flatten(2)
        B=B.flatten(2)
        batch=transformer.data_gen(B,Inx)
        y = self.transformer(batch.src1,batch.trg1,batch.src_mask,batch.trg_mask)
        y=torch.cat((B[:,0:1,:,],y),dim=1)
        for i in range(times):
            ouy=y[:,i,:].squeeze(1)
            Y=self.patch_decoding(ouy)
            Y=Y.unsqueeze(-1)
            if i==0:
                out=Y
                continue
            out=torch.cat((out,Y),dim=-1)
        out=out.reshape(original_shape)

        return out
    
    def train_enc(self,x):
        original_shape = x.shape
        times=x.shape[-1]
        for i in range(times):
            inx=x[:,:,:,:,:,i].squeeze(-1)
            inx = self.patch_embedding(inx).unsqueeze(1)
            if i==0:
                Inx=inx
                continue
            Inx=torch.concat((Inx,inx),dim=1)
        for i in range(times):
            ouy=Inx[:,i,:].squeeze(1)
            Y=self.patch_decoding(ouy)
            Y=Y.unsqueeze(-1)
            if i==0:
                out=Y
                continue
            out=torch.cat((out,Y),dim=-1)
        out=out.reshape(original_shape)
        return out


class RcnBlock(nn.Module):
    def __init__(self,layer_num,ker_size,in_channnel,out_channel ):
        super(RcnBlock,self).__init__()
        self.in_layer=nn.Conv3d(in_channels=in_channnel+out_channel,out_channels=2*(in_channnel+out_channel),kernel_size=ker_size,padding='same')
        self.out_layer=nn.Conv3d(in_channels=2*(in_channnel+out_channel),out_channels=out_channel,kernel_size=ker_size,padding='same')
        if layer_num>2:
            self.layers=nn.ModuleList([nn.Conv3d(in_channels=2*(in_channnel+out_channel),out_channels=2*(in_channnel+out_channel),kernel_size=ker_size,padding='same') for _ in range(layer_num-2)])
        else:
            self.layers=nn.ModuleList([nn.Identity])
        self.act=nn.Tanh()
            
    def forward(self,x,last):
        input=torch.cat((x,last),dim=1)
        inner=self.in_layer(input)
        for i in range(len(self.layers)):
            inner=self.layers[i](self.act(inner))
        out=self.out_layer(inner)
        return out
        

class Rcn(nn.Module):
    def __init__(self,layers,ker_size,in_channnel_x,in_channle_b,out_channel,simple_Train=False):
        super(Rcn,self).__init__()
        self.block=RcnBlock(layers,ker_size,in_channnel_x+in_channle_b,out_channel)
        self.simple=simple_Train
        
    def forward(self,x,b,y):
        out=y.clone()
        times=x.shape[1]
        if self.simple:
            for i in range(times):
                xi=x[:,i,:,:,:,:].squeeze(1)
                bi=b[:,i,:,:,:,:].squeeze(1)
                yi=y[:,i,:,:,:,:].squeeze(1)
                input=torch.cat((xi,bi),dim=1)
                temp=self.block(input,yi)
                out[:,i,:,:,:,:]=temp
        else:
            for i in range(times):
                xi=x[:,i,:,:,:,:].squeeze(1)
                bi=b[:,i,:,:,:,:].squeeze(1)
                input=torch.cat((xi,bi),dim=1)
                temp=self.block(input,temp)
                out[:,i,:,:,:,:]=temp.unsequeeze(dim=1)
        return out
    
    def set_simple_train(self,label:bool):
        self.simple=label
                

# class BigCNN3D(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=51,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias=True,
#                  padding_mode='zeros',
#                  device=None,
#                  dtype=None):
#         super(BigCNN3D,self).__init__()
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.kernel_size=kernel_size
#         self.stride=stride
#         self.padding=padding
#         self.dilation=dilation
#         self.groups=groups
#         self.need_bias=bias
#         self.padding_mode=padding_mode
#         self.device=device
#         self.dtype=dtype
#         self.outloop=nn.Parameter(torch.Tensor(out_channels,in_channels//groups,1,1,1))
#         self.midloop = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, 1, 1, 1))
#         self.innerloop = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, 1, 1, 1))
#         self.bias=torch.zeros((out_channels))
#         if self.need_bias==True:
#             self.bias=nn.Parameter(torch.Tensor(out_channels))

#     def forward(self,x):
#         outkernel=self.outloop.repeat(1,1,self.kernel_size,self.kernel_size,self.kernel_size)
#         midkernel=self.midloop.repeat(1,1,self.kernel_size//4*2+1,self.kernel_size//4*2+1,self.kernel_size//4*2+1)
#         innerkernel=self.innerloop.repeat(1,1,1,1,1)
#         kernel=outkernel
#         kernel[:,:,(self.kernel_size-(self.kernel_size//4*2+1))//2:-(self.kernel_size-(self.kernel_size//4*2+1))//2,(self.kernel_size-(self.kernel_size//4*2+1))//2:-(self.kernel_size-(self.kernel_size//4*2+1))//2,(self.kernel_size-(self.kernel_size//4*2+1))//2:-(self.kernel_size-(self.kernel_size//4*2+1))//2]=midkernel
#         kernel[:,:,self.kernel_size//2:self.kernel_size//2,self.kernel_size//2:self.kernel_size//2,self.kernel_size//2:self.kernel_size//2]=innerkernel
#         out=nn.functional.conv3d(x,kernel,bias=self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
#         return out

# class BigCnnNet(nn.Module):
#     def __init__(self,layers):
#         super(BigCnnNet,self).__init__()
#         self.encodes=nn.ModuleList(
#             [
#                 BigCNN3D(3,3,kernel_size=5,padding='same') for _ in range(layers)
#                 # nn.Conv3d(3, 3, kernel_size=11, padding='same') for _ in range(layers)
#             ]
#         )
#         self.cal=Trans_new(3,3,16,8,4,1)
#         self.decodes = nn.ModuleList(
#             [
#                 BigCNN3D(3, 3, kernel_size=5,padding='same') for _ in range(layers)
# #                 nn.Conv3d(3, 3, kernel_size=11, padding='same') for _ in range(layers)
#             ]
#         )
#     def forward(self,x,bond):
#         # print(f'time:{time.time()}start encode')
#         B,C,X,Y,Z,T=x.shape
#         x=x.permute(0,-1,1,2,3,4).contiguous().view(B*T,C,X,Y,Z)
#         bond = bond.permute(0, -1, 1, 2, 3, 4).contiguous().view(B*T, C , X, Y, Z)
#         for layer in self.encodes:
#             x=x+layer(x)
#             bond=bond+layer(bond)
# #         print(f'time:{time.time()}calculating')
#         x=x.view(B,T,C,X,Y,Z).permute(0,2,3,4,5,1)
#         bond = bond.view(B,T,C,X,Y,Z).permute(0,2,3,4,5,1)
#         # out=self.cal(x,bond)
# #         print(f'time:{time.time()}start decode')
#         out=x.permute(0, -1, 1, 2, 3, 4).contiguous().view(B*T, C , X, Y, Z)
#         for layer in self.decodes:
#             out=layer(out)
#         out = out.view(B,T,C,X,Y,Z).permute(0,2,3,4,5,1)
# #         print(f'time:{time.time()}EOC')
#         return out



# # 定义网络参数
# in_channels = 3
# out_channels = 3
# patch_size = 16
# embedding_dim = 16
# num_heads = 8
# num_layers = 6

# # 创建网络实例
# net = Trans(12288,8,1).cuda()
# # # network = CNN4D(num_layers)

# # # 示例输入
# input_tensor1 = torch.randn(3,64,12288).cuda()
# input_tensor2 = torch.randn(3,64,12288).cuda()
# o= torch.randn(3,64,12288).cuda()

# # # 前向传播
# # output_tensor = network(input_tensor1,input_tensor2)
# # print(output_tensor.shape)  # 输出应为 (batch, channel, X, Y, Z, time)

# # # net=BigCNN3D(3,3,padding='same').cuda()
# optim=torch.optim.Adam(net.parameters())
# correction=nn.MSELoss()
# # input_tensor = torch.randn(2, 3, 64, 64, 64).cuda()
# for i in range(10):
#     optim.zero_grad()
#     out=net(input_tensor1,input_tensor2)
#     loss=correction(out,o)
#     loss.backward()
#     optim.step()

# input_ex=(torch.randn(1,3,64,64,64,64,device='cuda:0'),torch.randn(1,3,64,64,64,64,device='cuda:0'))
# model=Trans_CNN4D(3,3,8,8,8,3).cuda()
# flops,params=profile(model,input_ex)
# flops, params = clever_format([flops, params], '%.3f')
# print(f"运算量：{flops}, 参数量：{params}")

# coder = Trans_CNN4D(3,3,8,6)
# coder.load_state_dict(torch.load("./model/Trans_CNN4D_[3, 3, 8, 6, 8, 1, True]_best.pt"),strict=False)
# coder.eval()
# coder.cuda()
# input=torch.rand((1,3,64,64,64,64),device='cuda')
# outpuy=coder(input)
# print(outpuy.shape)
# mean=(outpuy-input).cpu().max()
# print(mean)
