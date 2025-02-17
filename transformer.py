import copy
import torch
import torch.nn as nn
import math
# from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = 'cuda:0'
import numpy as np
import time



class PatchEmbed(nn.Module):   # input (b,c,h,w,t)
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
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size, img_size)  # 224 -> (224, 224)
        patch_size = (patch_size, patch_size, patch_size)  # 16 -> (16, 16)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])  # 计算原始图像被划分为(14, 14)个小块
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]  # 计算patch的个数为14*14=196个
        # 定义卷积层
        self.proj = nn.Conv3d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size) 
        # 定义归一化方式
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        此函数用于前向传播
        :param x: 原始图像
        :return: 处理后的图像
        """
        # 对图像依次作卷积、展平和调换处理: [B, C, H, W, T] -> [B, C, HWT] -> [B, HWT, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 归一化处理
        x = self.norm(x)
        return x


class PatchDecoder(nn.Module):   # input (b,c,h,w,t)
    """
    对3D图像作Patch decoder操作
    """
    def __init__(self, img_size=32, patch_size=8, in_c=256, out_dim=16, norm_layer=None):
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
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])  # 计算原始图像被划分为(14, 14)个小块
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]  # 计算patch的个数为14*14=196个
        # 定义卷积层
        # self.proj = nn.Conv3d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.proj = nn.ConvTranspose3d(in_channels=in_c, out_channels=out_dim, kernel_size=patch_size, stride=patch_size) 

        # 定义归一化方式
        self.norm = norm_layer(out_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        此函数用于前向传播
        :param x: 原始图像
        :return: 处理后的图像
        """
        x = x.transpose(1, 2).reshape([-1, 256, self.patch_num, self.patch_num, self.patch_num])

        # 对图像依次作卷积、展平和调换处理: [B, C, H, W, T] -> [B, C, HWT] -> [B, HWT, C]
        x = self.proj(x)
        # 归一化处理
        x = self.norm(x)
        return x


# inputdata = torch.randn(size=(5, 4, 32, 32, 32)).cuda()
#
# Emodel = PatchEmbed()
# y = Emodel(inputdata)
# Dmodel = PatchDecoder()
# z = Dmodel(y)
# print(y)

class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器架构。这个和许多的基础其他型号
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """接收并处理掩蔽的 src 和 target 序列。"""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Transformer(nn.Module):
    def __init__(self, d_model, head_num, N, ff_d, dropout, tgt_vocab=256):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(head_num, d_model) 
        ff = PositionwiseFeedForward(d_model, ff_d, dropout) 
        position = PositionalEncoding(d_model, dropout) 
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout) , N)   # 编码端，论文中包含了6个Encoder模块
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout) , N)   # 解码端，也是6个Decoder模块
        self.src_embed = nn.Sequential(
                                       # PatchEmbed(img_size=32, patch_size=8, in_c=4, embed_dim=d_model),
                                       # Embeddings(d_model, src_vocab) ,
                                       c(position))  # 输入Embedding模块
        self.tgt_embed = nn.Sequential(
                                       # PatchEmbed(img_size=32, patch_size=8, in_c=4, embed_dim=d_model),
                                       # Embeddings(d_model, tgt_vocab) ,
                                       c(position))  # 输出Embedding模块
        # data = torch.from_numpy(np.random.randint(1, 10, size=(5, 64)))
        # data[:, 0] = 1
        # src = Variable(data, requires_grad=False)
        # tgt = Variable(data, requires_grad=False)
        # self.Batch = Batch(src, tgt, 0)

        self.generator = Generator(d_model, tgt_vocab)  # 最终的Generator层，包括Linear+softmax

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # self.src_embed(src)
        # mask = Batch(src, tgt)
        # encoder的结果作为decoder的memory参数传入，进行decode
        code = self.encode(src, src_mask)
        return self.decode(code, src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """产生 N 个相同的层。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续encode N次(这里为6次)
        这里的Encoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """ 构造层范数模块（详见引文）."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory
        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)
    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = scores.cpu()
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 将mask后的attention矩阵按照最后一个维度进行softmax，归一化到0~1
    p_attn = F.softmax(scores, dim=-1)
    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    # p_attn = p_attn.cuda()
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # h为head数量，保证可以整除，论文中该值是8
        assert d_model % h == 0
        # 得到一个head的attention表示维度，论文中是512/8=64
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):  # max_len=256
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号），已经忘记中学对数操作的同学请自行补课哈
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(N=6, d_model=256, d_ff=2048, h=8, dropout=0.1):
    model = Transformer(d_model, head_num=h, N=N, ff_d=d_ff, dropout=dropout) 
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model 


# inputdata = torch.randn(size=(5, 4, 32, 32, 32))
# tmp_model = make_model(10, 10, 2)
# y = tmp_model(inputdata)
# print(y)



def subsequent_mask(size):
    " 掩盖后续 positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



def run_epoch(data_iter, model, loss_compute):
    """ 标准训练和记录功能"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        recon, code = model.forward(batch.src, batch.trg,
                                    batch.src_mask, batch.trg_mask)
        loss = loss_compute(recon, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens = batch.ntokens
        tokens = batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOpt:
    """实现rate的 Optim 包装器。"""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """ 更新参数和rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """ 在上面实现`lrate`"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """ 实施标签平滑."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.L1Loss()  #nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # target = target.cuda()
        # true_dist = x.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        # index = target.data.to(dtype=torch.int64)
        # true_dist.scatter_(1, index.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # if mask.dim() > 0:
        #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # self.true_dist = true_dist
        out = self.criterion(x, target)
        return out  #Variable(true_dist, requires_grad=False))


# def data_gen(batch, u1, u2, u3, u4, u5):
#     """ 为 src-tgt 复制任务生成随机数据。"""
#     sampnum = u1.size()[0]
#     begin = torch.ones(size=(sampnum, 1, 256)) 
#     u1 = torch.cat((begin, u1), dim=1)
#     u2 = torch.cat((begin, u2), dim=1)
#     u3 = torch.cat((begin, u3), dim=1)
#     u4 = torch.cat((begin, u4), dim=1)
#     u5 = torch.cat((begin, u5), dim=1)
#     nbatches = sampnum//batch
#     for i in range(nbatches):
#         src1 = Variable(u1[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         tgt1 = Variable(u1[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         src2 = Variable(u2[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         tgt2 = Variable(u2[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         src3 = Variable(u3[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         tgt3 = Variable(u3[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         src4 = Variable(u4[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         tgt4 = Variable(u4[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         src5 = Variable(u5[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         tgt5 = Variable(u5[i * batch:(i + 1) * batch, :, :], requires_grad=False)
#         yield Batch(src1, tgt1, src2, tgt2, src3, tgt3, src4, tgt4, src5, tgt5, )

def data_gen(condition_in, u):
    """ 为 src-tgt 复制任务生成随机数据。"""

    src1 = Variable(condition_in, requires_grad=False)
    tgt1 = Variable(u, requires_grad=False)

    return Batch(src1, tgt1)



class Batch:
    """用于在训练期间使用掩码保存一批数据的对象。"""
    def __init__(self, src1, tgt1):
        self.src1 = src1
        self.trg1 = tgt1[:, :-1, :]  # trg[:, :-1]
        self.trg1_y = tgt1[:, 1:, :]  # trg[:, 1:]

        self.src_mask = torch.full((src1.size()[0], 1, src1.size()[1]), True, dtype=torch.bool,device=src1.device)
        self.trg_mask = self.make_std_mask(self.trg1)
        # a = self.trg_y.size()[0]*self.trg_y.size()[1]
        self.ntokens = torch.tensor(self.trg1_y.size()[0]*self.trg1_y.size()[1],device=self.trg1_y.device) 

    @staticmethod
    def make_std_mask(tgt):
        """创建一个掩码来隐藏填充和未来的单词。"""
        tgt_mask = torch.full((tgt.size()[0], 1, tgt.size()[1]), True, dtype=torch.bool,device=tgt.device)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))
        return tgt_mask




class SimpleLossCompute:
    """ 一个简单的损失计算和训练函数。"""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1, x.size(-1))) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # a = loss.data[0]
        # b = norm
        return loss.data * norm


# 训练简单的复制任务。
# V = 65
# inputdata = torch.randn(size=(30, 4, 32, 32, 32))
# patchembed = PatchEmbed(img_size=32, patch_size=8, in_c=4, embed_dim=256)
# code = patchembed(inputdata)
#
#
# criterion = LabelSmoothing(size=256, padding_idx=0, smoothing=0.0)
# model = make_model(N=2)  # N 串联数量
# model_opt = NoamOpt(256, 1, 400,
#                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#
# for epoch in range(10):
#     model.train()
#     run_epoch(data_gen(10, code), model,
#               SimpleLossCompute(model.generator, criterion, model_opt))
    # model.eval()
    # print(run_epoch(data_gen(30, code), model,
    #                 SimpleLossCompute(model.generator, criterion, None)))




#
#
# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     memory = model.encode(src, src_mask)
#     #ys是decode的时候起始标志
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
#     print(ys)
#     for i in range(max_len-1):
#         out = model.decode(memory, src_mask,
#                            Variable(ys),
#                            Variable(subsequent_mask(ys.size(1))
#                                     .type_as(src.data)))
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim = 1)
#         next_word= next_word.data[0]
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#         print("ys:"+str(ys))
#
#     return ys
#
# model.eval()
# src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
# src_mask = Variable(torch.ones(1, 1, 10) )
# # print("ys:"+str(ys))
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
