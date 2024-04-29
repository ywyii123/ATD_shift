
from basicsr.utils.registry import ARCH_REGISTRY

import torch
import torch.nn as nn
import math
from basicsr.archs.arch_util import NAFBlock, to_2tuple, trunc_normal_


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

        # self.layers = nn.ModuleList()
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

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ):
        super(ResidualBlock, self).__init__()


        # main_path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # skip connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut(x)
        # out = nn.ReLU(inplace=True)(out)
        return out

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
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                # num_feat = num_feat // 4
                # m.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
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
                # m.append(nn.Conv2d(chns, chns, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))
                m.append(nn.Conv2d(chns * 4, chns, 3, 1, 1))
                chns = chns * 2
        elif scale == 3:
            # m.append(nn.Conv2d(chns, chns, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))
            m.append(nn.Conv2d(chns * 9, chns, 3, 1, 1))
        
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


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 repeat_times, 
                 emb_dim, 
                 num_blocks, 
                 upscale,
                 gt_size,
                 window_size,
                 mlp_ratio,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 downscale=1,
                 convffn_kernel_size=5,
                 qkv_bias=True,
                 channel_mult_emb=4,
                 time_emb=False,
                 cond_lq=False,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 upsampler='pixelshuffle',
                 interpolation=None,
                 block_type='edsr',
                 use_conv=True,
                 res=False,
                 **kwargs,
                 ):
        super(EDSR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_type = block_type
        self.gt_size = gt_size
        num_in_channels = in_channels
        self.upscale = upscale
        self.downscale = downscale
        self.cond_lq = cond_lq
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.upsampler = upsampler
        self.res = res
        self.repeat_times = repeat_times

        num_feat = emb_dim
        if interpolation is not None:
            # assert upsampler == 'interpolation'
            self.interpolation = interpolation
        if time_emb:
            time_embed_dim = emb_dim * channel_mult_emb
            self.time_embed = nn.Sequential(
                nn.Linear(emb_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        num_in_channels = in_channels * repeat_times
        if cond_lq:
            num_in_channels = in_channels + num_in_channels
        
        self.conv0 = nn.Conv2d(num_in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
            
        img_size = math.ceil(gt_size / self.upscale)
        self.layers = nn.ModuleList()
        if block_type == 'swin':
            relative_position_index_SA = self.calculate_rpi_sa()
            self.register_buffer('relative_position_index_SA', relative_position_index_SA)
            self.num_layers = len(depths)         
            for i_layer in range(self.num_layers):
                layer = BasicBlock(
                    dim=emb_dim,
                    input_resolution=(img_size, img_size),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                self.layers.append(layer)
            self.norm = norm_layer(emb_dim)
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=1,
                in_chans=emb_dim,
                embed_dim=emb_dim,
                norm_layer=norm_layer if self.patch_norm else None)

            self.patch_unembed = PatchUnEmbed(
                img_size=img_size,
                patch_size=1,
                in_chans=emb_dim,
                embed_dim=emb_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        elif block_type == 'edsr':
            self.block = ResidualBlock(emb_dim, emb_dim,)
            for i in range(num_blocks):
                self.layers.append(
                    self.block
                )
        elif block_type == 'naf':
            self.block = NAFBlock(emb_dim,)
            for i in range(num_blocks):
                self.layers.append(
                    self.block
                )

        # self.layers = nn.Sequential(*[self.block for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(emb_dim, emb_dim, 3, 1, 1)

        # self.conv_before_upsample = nn.Sequential(
        # nn.Conv2d(emb_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        if self.upsampler == 'pixelshuffle':
            self.upsample = nn.Sequential(
                nn.PixelShuffle(upscale),
                nn.Conv2d(num_feat // (upscale ** 2), num_feat // (upscale ** 2), 3, 1, 1) 
                if use_conv else nn.Identity(),
                )
            num_feat = num_feat // (upscale ** 2)
            if downscale != 1:
                # self.conv_before_downsample = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.downsample = nn.Sequential(
                    # nn.Conv2d(num_feat, num_feat, kernel_size=downscale, stride=downscale, padding=0)
                    nn.PixelUnshuffle(downscale)
                )
            num_feat = num_feat * (downscale ** 2)
        elif self.upsampler == 'interpolation':
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
    def forward(self, x, lq=None, gt_size=None):
        x_size = (x.shape[2], x.shape[3])
        
        h_ori, w_ori = x_size
        if self.block_type == 'swin':
            mod = self.window_size * self.downscale
        else:
            mod = self.downscale
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
        # x = x.repeat(1, self.repeat_times, 1, 1)
        if lq != None and self.cond_lq:
            lq = torch.cat([lq, torch.flip(lq, [2])], 2)[:, :, :h, :]
            lq = torch.cat([lq, torch.flip(lq, [3])], 3)[:, :, :, :w]
            x = torch.cat([lq, x], dim=1)
        
        x_size = (h, w)

        if self.block_type == 'swin':
            attn_mask = self.calculate_mask([h, w]).to(x.device)
            params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        
        x_first = self.conv0(x)
        x_next = x_first

        if self.block_type == 'swin':
            x_next = self.patch_embed(x_next)
            for layer in self.layers:
                x_next = layer(x_next, x_size, params)
            x_next = self.norm(x_next)
            x_next = self.patch_unembed(x_next, x_size)
            out = self.conv_after_body(x_next) + x_first
            # out = self.conv_before_upsample(out)
        else:
            for layer in self.layers:
                x_next = layer(x_next,)
            out = self.conv_after_body(x_next) + x_first
            # out = self.conv_before_upsample(out)

        if self.upsampler == 'pixelshuffle':
            out = self.upsample(out)
            if self.downscale != 1:
                # out = self.conv_before_downsample(out)
                out = self.downsample(out)
            out = self.conv_last(out)
            
        elif self.upsampler == 'interpolation':
            sf = self.upscale / self.downscale
            out = self.conv_hr(self.conv_up1(torch.nn.functional.interpolate(out, scale_factor=sf, mode=self.interpolation)))
            # out = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(out, scale_factor=sf, mode=self.interpolation)))
            # out = self.conv_last(self.lrelu(self.conv_hr(out)))
            out = self.conv_last(out)

        

        if gt_size is not None:
            out = out[..., :gt_size, :gt_size]
        if self.res:
            sf = self.upscale / self.downscale
            lq_upsample = torch.nn.functional.interpolate(lq, scale_factor=sf, mode='bilinear')
            lq_upsample = lq_upsample[..., :gt_size, :gt_size]
            out = out + lq_upsample
        return out
    

@ARCH_REGISTRY.register()
class EDSRList(nn.Module):
    def __init__(self, **opt):
        super(EDSRList, self).__init__()
        self.opt = opt
        self.net_g_list = nn.ModuleList()
        for opt_net in opt.values():
            net_g = EDSR(
                in_channels=opt_net['in_channels'], 
                out_channels=opt_net['out_channels'],
                repeat_times=opt_net['repeat_times'],
                num_blocks=opt_net['num_blocks'],
                emb_dim=opt_net['emb_dim'],  
                upscale=opt_net['upscale'],
                downscale=opt_net['downscale'],
                gt_size=opt_net['gt_size'],
                window_size=opt_net['window_size'],
                mlp_ratio=opt_net['mlp_ratio'],
                depths=opt_net['depths'],
                num_heads=opt_net['num_heads'],
                qkv_bias=True,
                channel_mult_emb=opt_net['channel_mult_emb'],
                cond_lq=opt_net['cond_lq'],
                use_conv=opt_net['use_conv'],
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                upsampler=opt_net['upsampler'],
                interpolation=opt_net['interpolation'],
                block_type=opt_net['block_type'],
                res=opt_net['res'],
            )
            self.net_g_list.append(net_g)
    
    def forward(self, x, timestep, lq=None, gt_size=None):
        return self.net_g_list[timestep](x, lq=lq, gt_size=gt_size)
