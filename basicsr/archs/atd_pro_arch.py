'''
An official Pytorch impl of `Transcending the Limit of Local Window: 
Advanced Super-Resolution Transformer with Adaptive Token Dictionary`.

Arxiv: 'https://arxiv.org/abs/2401.08209'
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, NAFBlock
from fairscale.nn import checkpoint_wrapper
from basicsr.utils.basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
import torch.distributed as dist


from basicsr.utils.registry import ARCH_REGISTRY


# Shuffle operation for Categorization and UnCategorization operations.
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]                        # B x half
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops



class SwinBlock(nn.Module):
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 shift_size,
                 mlp_ratio,
                 convffn_kernel_size,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size

        
        self.wqkv = nn.Linear(dim, 3*dim, bias=qkv_bias)

        self.attn_win = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=act_layer)
    
    def forward(self, x, x_size, params):        
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        # SW-MSA
        qkv = qkv.reshape(b, h, w, c3)

        # cyclic shift
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_qkv, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn_win(x_windows, rpi=params['rpi_sa'], mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x
        x = shortcut + x_win.view(b, n, c)
        # FFN
        x = x + self.convffn(self.norm2(x), x_size)
        return x




class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicBlock(nn.Module):
    """ A basic ATD Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=1,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        img_size = self.input_resolution[0]
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                SwinBlock(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    convffn_kernel_size=convffn_kernel_size,
                    norm_layer=norm_layer,
                    qkv_bias=qkv_bias
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Token Dictionary
        # self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params):
        x0 = x
        # td = self.td.repeat([b, 1, 1])
        for layer in self.layers:
            x = layer(x, x_size, params)
            
        if self.downsample is not None:
            x = self.downsample(x)
        return x0 + self.patch_embed(self.conv(self.patch_unembed(x, x_size)))

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops





class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                
                # m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                num_feat = num_feat // 4
                m.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
            # num_feat = num_feat // 9
            # m.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat,):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        chns = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(chns, chns, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
                chns = chns * 2
        elif scale == 3:
            # m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.Conv2d(chns, chns, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
        
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class UpAndDownBeforeOut(nn.Module):
    def __init__(self,
                 num_feat,
                 upscale,
                 downscale,
                 outchns,
                 upsampler='pixelshuffle',
                 last_layers=None,
                 interpolation=None
                 ):
        super(UpAndDownBeforeOut, self).__init__()
        self.upsampler = upsampler
        self.num_feat = num_feat
        self.upscale = upscale
        self.downscale = downscale
        self.last_layers = last_layers
        if interpolation is not None:
            self.interpolation = interpolation
        
            
        if upsampler == 'pixelshuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale),
            )
            num_feat = num_feat
            
            if downscale != 1:
                # self.conv_before_downsample = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                # self.conv_last = nn.Conv2d(num_feat, outchns, kernel_size=downscale, stride=downscale, padding=0)
                self.downsample = nn.Sequential(
                    # nn.Identity()
                    # nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                    # nn.Conv2d(num_feat, num_feat, kernel_size=downscale, stride=downscale, padding=0),
                    nn.PixelUnshuffle(downscale),
                    # nn.Conv2d(num_feat * (downscale ** 2), num_feat, 3, 1, 1)
                )
            num_feat = num_feat * (downscale ** 2)

        elif upsampler == 'interpolation':
            scale = upscale / downscale
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if scale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            
        self.conv_last = nn.Conv2d(num_feat, outchns, 3, 1, 1)

    
    def forward(self, x, emb=None):
        if self.upsampler == 'pixelshuffle':
            # x = self.conv_before_upsample(x)
            x = self.upsample(x)
            if self.downscale != 1:
                # x = self.conv_before_downsample(x)
                x = self.downsample(x)
            if self.last_layers is not None and emb is not None:
                for layer in self.last_layers:
                    x = layer(x, emb)
            x = self.conv_last(x)
        elif self.upsampler == 'interpolation':
            # x = self.conv_before_upsample(x)
            sf = self.upscale / self.downscale
            if sf == 4:
                x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode=self.interpolation,)))
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode=self.interpolation,)))
            else:
                x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=sf, mode=self.interpolation,)))
            if self.last_layers is not None and emb is not None:
                for layer in self.last_layers:
                    x = layer(x, emb) 
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, 
                 emb_dim,
                 depth,
                 time_embed_dim,
                 num_heads=None,
                 window_size=None,
                 block_type='naf',
                 dropout=0.1,
                 convffn_kernel_size=5,
                 mlp_ratio=4.
                 ):
        super(BasicLayer, self).__init__()
        self.block_type = block_type
        self.layers = nn.ModuleList([])
        self.time_emb_block_0 = ResBlock(
                channels=emb_dim,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=emb_dim,
                use_scale_shift_norm=True,)
        
        for _ in range(depth):
            if block_type == 'naf':
                self.layers.append(
                    NAFBlock(
                        c=emb_dim,
                    )
                )
            elif block_type == 'edsr':
                self.layers.append(
                    EDSRBlock(
                        in_channels=emb_dim,
                        out_channels=emb_dim,
                    )
                )
        
        if block_type == 'swin':
            self.layers.append(
                SwinLayer(
                    dim=emb_dim,
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio
                )
            )
            
        # self.time_emb_block_1 = ResBlock(
        #         channels=emb_dim,
        #         emb_channels=time_embed_dim,
        #         dropout=dropout,
        #         out_channels=emb_dim,
        #         use_scale_shift_norm=True,)
        

    def forward(self, x, emb, x_size=None, params=None):
        x = self.time_emb_block_0(x, emb)
        for layer in self.layers:            
            if self.block_type == 'swin':
                x = layer(x, x_size, params)
            else:
                x = layer(x)
        
        # x = self.time_emb_block_1(x, emb)
        return x


class SwinLayer(nn.Module):
    """ A basic ATD Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 convffn_kernel_size=5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=1,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = img_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                SwinBlock(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    convffn_kernel_size=convffn_kernel_size,
                    norm_layer=norm_layer,
                    qkv_bias=qkv_bias
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample((img_size, img_size), dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Token Dictionary
        # self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params):
        x0 = x
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size, params)
        x = self.patch_unembed(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x0 + self.conv(x)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops



class EDSRBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ):
        super(EDSRBlock, self).__init__()


        # main_path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # skip connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut(x)
        # out = nn.ReLU(inplace=True)(out)
        return out

@ARCH_REGISTRY.register()
class ATD_pro(nn.Module):
    r""" ATD
        A PyTorch impl of : `Transcending the Limit of Local Window: Advanced Super-Resolution Transformer 
                             with Adaptive Token Dictionary`.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 gt_size=256,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 channel_mult_emb=4,
                 cond_lq=False,
                 dropout=0.1,
                 time_emb=False,
                 block_type='edsr',
                 up_list=[],
                 down_list=[],
                 encoder_list=[],
                 decoder_list=[],
                 interpolation=None,
                 res=False,
                 num_last_layers=0,
                #  repeat_list=[],
                 **kwargs):
        super().__init__()
        self.cond_lq = cond_lq
        # num_in_ch = in_chans

        # if cond_lq:
        #     num_in_ch = in_chans * 2
        num_out_ch = in_chans
        self.block_type = block_type
        num_feat = embed_dim
        self.img_range = img_range
        self.num_last_layers = num_last_layers
        self.res = res
        # if in_chans == 3:
        #     rgb_mean = (0.4488, 0.4371, 0.4040)
        #     self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # else:
        #     self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        
        self.up_list = up_list
        self.down_list = down_list

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        if time_emb:
            time_embed_dim = embed_dim * channel_mult_emb
            self.time_embed = nn.Sequential(
                nn.Linear(embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        self.encoders = nn.ModuleList()
        for scale in encoder_list:
            in_chns = in_chans * scale ** 2
            if cond_lq:
                in_chns += in_chans
            encoder = nn.Conv2d(in_chns, embed_dim, kernel_size=3, stride=1, padding=1)
            self.encoders.append(encoder)
        # for repeat_times in repeat_list:
        #     in_chns = in_chans * repeat_times
        #     if cond_lq:
        #         in_chns += in_chans
        #     encoder = nn.Conv2d(in_chns, embed_dim, kernel_size=3, stride=1, padding=1)
        #     self.encoder_list.append(encoder)

        if block_type == 'swin':

            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=embed_dim,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

            # merge non-overlapping patches into image
            self.patch_unembed = PatchUnEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=embed_dim,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)

            # absolute position embedding
            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)

            # relative position index
            relative_position_index_SA = self.calculate_rpi_sa()
            self.register_buffer('relative_position_index_SA', relative_position_index_SA)

            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    emb_dim=embed_dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    time_embed_dim=time_embed_dim,
                    block_type=block_type,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                self.layers.append(layer)
            # self.norm = norm_layer(self.num_features)
        else:
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                self.layers.append(
                    BasicLayer(
                        emb_dim=embed_dim,
                        depth=depths[i_layer],
                        time_embed_dim=time_embed_dim,
                        block_type=block_type,
                        dropout=dropout
                    )
                )


        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # self.encoder_list = nn.ModuleList()
        # self.decoder_list = nn.ModuleList()


        # self.up_down_list = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for scale in decoder_list:
            self.decoders.append(nn.Sequential(
                nn.PixelShuffle(scale),
                nn.Conv2d(embed_dim // (scale ** 2), num_out_ch, 3, 1, 1)
            )
            )


        if num_last_layers != 0:
            last_layers = nn.ModuleList()
            for i in range(num_last_layers):
                last_layer = BasicLayer(
                    emb_dim=embed_dim,
                    depth=depths[0],
                    time_embed_dim=time_embed_dim,
                    block_type=block_type,
                    dropout=dropout
                )
                last_layers.append(last_layer)
        else:
            last_layers = None

        # for upscale, downscale in zip(up_list, down_list):
        #     up_down_model = UpAndDownBeforeOut(
        #         num_feat=num_feat,
        #         upscale=upscale,
        #         downscale=downscale,
        #         outchns=num_out_ch,
        #         upsampler=upsampler,
        #         interpolation=interpolation,
        #         last_layers=last_layers
        #     )
        #     self.up_down_list.append(up_down_model)

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params, emb=None):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        for layer in self.layers:
            x = layer(x, emb, x_size, params)
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x
    
    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index
    
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, timestep, sigma, lq=None, gt_size=None):
        x_size = (x.shape[2], x.shape[3])
        h_ori, w_ori = x_size
        # downscale = self.down_list[timestep]
        # upscale = self.up_list[timestep]
        encoder = self.encoders[timestep]
        decoder = self.decoders[timestep]
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_noise = (sigma).log() / 4
        emb = self.time_embed(timestep_embedding(c_noise.flatten(), self.embed_dim)).type(x.dtype)

        if self.block_type == 'swin':
            mod = self.window_size
            h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
            w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
            h, w = h_ori + h_pad, w_ori + w_pad
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
            if lq is not None and self.cond_lq:
                lq = torch.cat([lq, torch.flip(lq, [2])], 2)[:, :, :h, :]
                lq = torch.cat([lq, torch.flip(lq, [3])], 3)[:, :, :, :w]
                x = torch.cat([lq, x], dim=1)
            x_size = (h, w)
            attn_mask = self.calculate_mask([h, w]).to(x.device)
            params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        else:
            (h, w) = x_size
            if lq is not None and self.cond_lq:
                lq = torch.cat([lq, torch.flip(lq, [2])], 2)[:, :, :h, :]
                lq = torch.cat([lq, torch.flip(lq, [3])], 3)[:, :, :, :w]
                x = torch.cat([lq, x], dim=1)

        # x_first = self.conv_first(x)
        # x_first = self.encoder_list[timestep](x)
        x_first = encoder(x)
        x_next = x_first

        if self.block_type == 'swin':
            for layer in self.layers:
                x_next = layer(x_next, emb, x_size=x_size, params=params)
        else:
            for layer in self.layers:
                x_next = layer(x_next, emb)
        out = self.conv_after_body(x_next) + x_first
        # if self.num_last_layers != 0:
        #     out = self.up_down_list[timestep](out, emb)
        # else:
        #     out = self.up_down_list[timestep](out,)
        out = decoder(out)


        if gt_size is not None:
            out = out[..., :gt_size, :gt_size]
        
        if self.res:
            sf = upscale / downscale
            lq_upsample = torch.nn.functional.interpolate(lq, scale_factor=sf, mode='bilinear')
            lq_upsample = lq_upsample[..., :gt_size, :gt_size]
            out = out + lq_upsample
        return out



    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)

        return flops
    

# class Encoder(nn.Module):
#     def __init__(
#             self,
#             upscale,
#             downscale,
#             in_chans,
#             repeat_times,
#             cond_lq,
#             emb_dim,
#     ):
#         super().__init__()
#         self.cond_lq = cond_lq
#         self.upscale = upscale
#         self.downscale = downscale
#         num_in_chns = in_chans * repeat_times
#         if cond_lq:
#             num_in_chns += 3
#         self.conv_first = nn.Conv2d(num_in_chns, emb_dim, 3, 1, 1)
#         self.downsample = nn.Sequential(nn.PixelUnshuffle(downscale))
#         self.upsample = nn.Sequential(nn.PixelShuffle(upscale))
    
#     def forward(self, x, lq=None):
#         # x = x.repeat(1, 3, 1, 1)
#         if lq is not None and self.cond_lq:
#             x = torch.cat([x, lq], dim=1)
#         x = self.conv_first(x)
#         return x
    



# class Decoder(nn.Module):
#     def __init__(
#             self,
#             upscale,
#             downscale,
#             in_chans,
#             repeat_times,
#             cond_lq,
#             emb_dim,
#     ):
#         super().__init__()
#         self.cond_lq = cond_lq
#         self.upscale = upscale
#         self.downscale = downscale
#         num_in_chns = in_chans * repeat_times
#         if cond_lq:
#             num_in_chns += 3
#         self.conv_first = nn.Conv2d(num_in_chns, emb_dim, 3, 1, 1)
    
#     def forward(self, x, lq=None):
#         # x = x.repeat(1, 3, 1, 1)
#         if lq is not None and self.cond_lq:
#             x = torch.cat([x, lq], dim=1)
#         x = self.conv_first(x)
#         return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )


        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

if __name__ == '__main__':
    upscale = 4

    model = ATD_pro(
        upscale=4,
        img_size=64,
        embed_dim=210,
        depths=[6, 6, 6, 6, 6, 6, ],
        num_heads=[6, 6, 6, 6, 6, 6, ],
        window_size=16,
        category_size=128,
        num_tokens=128,
        reducted_dim=20,
        convffn_kernel_size=5,
        img_range=1.,
        down_c=20,
        mlp_ratio=2,
        upsampler='pixelshuffle')

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    print(128, 128, model.flops([128, 128]) / 1e9, 'G')
    print(256, 256, model.flops([256, 256]) / 1e9, 'G')

    # Test
    _input = torch.randn([2, 3, 64, 64])
    output = model(_input)
    print(output.shape)