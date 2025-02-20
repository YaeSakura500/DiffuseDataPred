from typing import Optional
import torch
import torch.nn as nn

class transformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 dim_feedforward=1024, 
                 dropout=0.1, 
                 activation='relu', 
                 custom_encoder=None, 
                 custom_decoder=None, 
                 layer_norm_eps=1e-05, 
                 batch_first=False, 
                 norm_first=False, 
                 bias=True, 
                 device=None, 
                 dtype=None,
                 simple_train:bool=True,
                 pred_length: Optional[int] = None):
        super(transformer,self).__init__()
        self.trans=nn.Transformer(d_model,
                                  nhead,
                                  num_encoder_layers,
                                  num_decoder_layers,
                                  dim_feedforward,
                                  dropout,activation,
                                  custom_encoder,
                                  custom_decoder,
                                  layer_norm_eps,
                                  batch_first,
                                  norm_first,
                                  bias,device,
                                  dtype)
        self.simple=simple_train
        self.pred_length=pred_length
        # self.in_linear=nn.Linear(2*d_model,d_model)
        self.out_linear=nn.Linear(d_model,d_model)
        self.gap=nn.Parameter(torch.zeros(1,1,d_model),requires_grad=True)
        
    def step(self,src,tgt):
        out=self.trans(src,tgt)
        return self.out_linear(out)
    
    def forward(self,Bondary,Feild,Truth):
        shape=Feild.shape
        # print(shape)
        bondary=Bondary.flatten(2)
        feild=Feild.flatten(2)
        truth=Truth.flatten(2)
        if self.pred_length is None :
            self.pred_length=shape[1]
        # condition=torch.cat((bondary,feild),dim=-1)
        # condition=self.in_linear(condition)
        condition=bondary+feild
        if shape[0]>1:
            condition_seq=torch.cat((condition,self.gap.repeat(shape[0],1,1)),dim=1)
        else:
            condition_seq=torch.cat((condition,self.gap),dim=1)
        out=feild[::,0:1:,]
        if self.simple==True:
            for i in range(1,self.pred_length):
                condition_seq=torch.cat((condition_seq,truth[::,i-1:i:,]),dim=1)
                next=self.step(condition_seq,truth[::,0:i:,])[::,-1::,]
                out=torch.cat((out,next),dim=1)
                # print(out.shape)
        else:
            next=out
            for _ in range(1,self.pred_length):
                condition_seq=torch.cat((condition_seq,next),dim=1)
                next=self.step(condition_seq,out)[::,-1::,]
                out=torch.cat((out,next),dim=1)
        out_shape=list(shape)
        out_shape[1]=self.pred_length
        out=out.reshape(out_shape)
        ground=truth[::,0:self.pred_length:,].reshape(out_shape)
        return out,ground
    
    def set_simple_train(self,simple_train:bool):
        self.simple=simple_train

    def set_prediction_length(self,pred_length: Optional[int] = None):
        self.pred_length=pred_length
    
    def test(self,Bondary,Feild):
        shape=Feild.shape
        bondary=Bondary.flatten(2)
        feild=Feild.flatten(2)
        # condition=torch.cat((bondary,feild),dim=-1)
        # condition=self.in_linear(condition)
        condition=bondary+feild
        condition_seq=torch.cat((condition,self.gap.repeat(shape[1])),dim=1)
        out=[Feild[::,0:1:,]]
        for _ in range(1,self.pred_length+1):
            condition_seq=torch.cat((condition_seq,out[-1]),dim=1)
            next=self.step(condition_seq,out[-1])[::,-2:-1:,]
            out.append(next)
        out=torch.cat(out,dim=1).reshape(shape)
        return out
    
    
