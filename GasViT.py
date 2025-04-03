#by yuuri 2023.5
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(inplace=True)
    )

def dwconv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    padding = int((kernal_size - 1) / 2)
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, padding=padding, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(inplace=True)
    )
    
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, mip, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mip, inp, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        se = self.avg_pool(x).view(n, c)
        se = self.fc(se).view(n, c, 1, 1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * se.expand_as(identity)
        out = out * a_w * a_h

        return out
        
class MHAttention(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(inp, mip, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mip, inp, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        se = self.avg_pool(x).view(n, c)  
        se = self.fc(se).view(n, c, 1, 1)  
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * se.expand_as(identity)
        out = out * a_w * a_h

        return out

class MHFusionAttention(nn.Module):
    def __init__(self, inp, conv_kernels = [1,3,5,7]):
        super(MHFusionAttention, self).__init__()
        self.split_channel = inp//4

        self.conv_1 = nn.Conv2d(inp, self.split_channel, kernel_size=conv_kernels[0], stride=1,
                               padding=conv_kernels[0]// 2,groups=self.split_channel)
        self.conv_2 = nn.Conv2d(inp, self.split_channel, kernel_size=conv_kernels[1], stride=1,
                               padding=conv_kernels[1] // 2,groups=self.split_channel)
        self.conv_3 = nn.Conv2d(inp, self.split_channel, kernel_size=conv_kernels[2], stride=1,
                               padding=conv_kernels[2] // 2,groups=self.split_channel)
        self.conv_4 = nn.Conv2d(inp, self.split_channel, kernel_size=conv_kernels[3], stride=1,
                               padding=conv_kernels[3] // 2,groups=self.split_channel)

        self.attn = CoordAtt(self.split_channel,self.split_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        x1_attn = self.attn(x1)
        x2_attn = self.attn(x2)
        x3_attn = self.attn(x3)
        x4_attn = self.attn(x4)

        out = torch.cat((x1_attn,x2_attn,x3_attn,x4_attn),dim=1)

        return out


class GasConvBlock(nn.Module):

    def __init__(self, inp=3, oup=16, stride=1, expansion=4):
        super().__init__()

        hidden_dim = int(inp * expansion)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            MHFusionAttention(hidden_dim,hidden_dim),
            nn.Hardswish(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )


    def forward(self, x):
        x = self.conv(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()

        inner_dim = dim // heads

        self.heads = heads
        self.scale = inner_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class MhLSa(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()

        self.inner_dim = dim

        self.heads = heads

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 2 + self.heads, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv,'b p n (h c) -> b h c p n',h=self.heads)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.inner_dim//self.heads, self.inner_dim//self.heads], dim=2)

        context_scores = self.attend(q)
        context_vector = k * context_scores
        context_vector = torch.sum(context_vector,dim=-1,keepdim=True)
        out = F.relu(v) * context_vector.expand_as(v)

        out = rearrange(out, 'b h c p n -> b p n (h c)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, 8, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class GasTransformer(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MhLSa(dim, 8, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class GasViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.1):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = dwconv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = GasTransformer(dim, depth, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        # self.conv4 = conv_1x1_bn(dim + channel, channel)

    def forward(self, x):
        y1 = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # y2 = x.clone()
        # Global representations
        _, _, h, w = x.shape
        # x = x.reshape()
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = x + y1
        return x

  class GasViT(nn.Module):
    def __init__(self, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.downsample = nn.ModuleList([])
        self.downsample.append(GasConvBlock(3, channels[0], 2, expansion))
        self.downsample.append(GasConvBlock(channels[0], channels[1], 2, expansion))
        self.downsample.append(GasConvBlock(channels[1], channels[2], 2, expansion))
        self.downsample.append(GasConvBlock(channels[2], channels[3], 2, expansion))
        self.downsample.append(GasConvBlock(channels[3], channels[4], 2, expansion))

        self.vit = nn.ModuleList([])
        self.vit.append(GasViTBlock(dims[0], L[0], channels[2], kernel_size, patch_size, int(dims[0] * 2)))
        self.vit.append(GasViTBlock(dims[1], L[1], channels[3], kernel_size, patch_size, int(dims[1] * 4)))
        self.vit.append(GasViTBlock(dims[2], L[2], channels[4], kernel_size, patch_size, int(dims[2] * 4)))


    def forward(self, x):
        output = []
        # x = self.patch_embed(x)
        x = self.downsample[0](x)
        x = self.downsample[1](x)
        x = self.downsample[2](x)
        #layer1
        x1 = self.vit[0](x)
        output.append(x1)
        #layer2
        x2 = self.downsample[3](x1)
        x2 = self.vit[1](x2)
        output.append(x2)
        #layer3
        x3 = self.downsample[4](x2)
        x3 = self.vit[2](x3)
        output.append(x3)

        return tuple(output)


def GasViT_xs():
    dims = [64, 80, 96]
    channels = [16, 32, 48, 64, 80]
    return SmokeViT((256, 256), dims, channels)

def GasViT_s():
    dims = [72, 96, 120]
    channels = [24, 48, 72, 96, 120]
    return SmokeViT((256, 256), dims, channels)

def GasViT():
    dims = [144, 192, 240]
    channels = [32, 64, 96, 128, 160]
    return SmokeViT((256, 256), dims, channels)
  
