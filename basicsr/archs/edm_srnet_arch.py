from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np
import torch
from torch.nn.functional import silu

from utils.basic_ops import normalization
from utils.swin_transformer import BasicLayer
from torch import Tensor

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))
#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------


class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True, resolution=256,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, swin=False, swin_params=dict(),
    ):
        super().__init__()
        self.attention = attention
        self.swin = swin
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads and swin == False:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)
        elif attention and swin == True:
            # swin_params['in_chans'] = out_channels
            self.swin_layer = BasicLayer(img_size=resolution,
                                         in_chans=out_channels,
                                         norm_layer=normalization,
                                         **swin_params)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads and self.swin == False:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        elif self.attention and self.swin == True:
            x = self.swin_layer(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

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

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

    


class SRNet(torch.nn.Module):
    def __init__(self,
        img_resolution,
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.        
        cond_lq             = False,
        up_list             = [1, 1, 1, 1],
        down_list           = [4, 3, 2, 1],

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 16,            # Number of residual blocks per resolution.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        swin                = False,
        swin_params         = None,
        scale               = 4,
    ):
        assert embedding_type in ['fourier', 'positional']


        super().__init__()
        self.cond_lq = cond_lq
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, swin=swin, swin_params=swin_params
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # self.encoder = torch.nn.PixelUnshuffle(scale)
        # self.decoder = torch.nn.PixelShuffle(scale)
        # self.decoder = torch.nn.ModuleList(
        #     [UNetBlock(in_channels=model_channels, out_channels=model_channels, up=True, **block_kwargs),
        #      UNetBlock(in_channels=model_channels, out_channels=model_channels, up=True, **block_kwargs)]
        # )
        cin = in_channels
        if cond_lq:
            cin = cin + in_channels
        self.encoder_list = torch.nn.ModuleList()
        for i in range(len(up_list)):
            upscale = up_list[i]
            downscale = down_list[i]
            encoder = Encoder(in_channels=cin,
                              out_channels=model_channels,
                              emb_dim=model_channels,
                              upscale=upscale,
                              downscale=downscale)
            self.encoder_list.append(encoder)
        self.decoder = Decoder(model_channels=model_channels,
                               emb_channels=emb_channels,
                               out_channels=out_channels,
                               upscale=scale,
                               swin=swin,
                               swin_params=swin_params,
                               resample_filter=resample_filter,
                               dropout=dropout
                               )
        

        # cin = in_channels * (scale ** 2)
        
        # self.conv_first = Conv2d(in_channels=cin, out_channels=model_channels, kernel=3, **init)

        self.layers = torch.nn.ModuleList()
        for i in range(num_blocks):
            block = UNetBlock(in_channels=model_channels, out_channels=model_channels, attention=True, **block_kwargs)
            self.layers.append(block)
        # self.last_layers = torch.nn.Sequential(
        #     GroupNorm(num_channels=model_channels, eps=1e-6),
        #     Conv2d(in_channels=model_channels, out_channels=out_channels, kernel=3, **init_zero)
        # )
        # self.conv_last = Conv2d(in_channels=model_channels, out_channels=out_channels * (scale ** 2), kernel=3, **init_zero)

    def forward(self, x, noise_labels, timestep=None, lq=None, class_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        encoder = self.encoder_list[timestep]
        # x = self.encoder(x)
        if lq is not None and self.cond_lq:
            x = torch.cat([x, lq], dim=1)
        x = encoder(x)
        temp = x
        # x = self.conv_first(x)
        for layer in self.layers:
            x = layer(x, emb)
        x = x + temp
        # for layer in self.decoder:
        #     x = layer(x, emb)
        # x = self.decoder(x, emb)
        x = self.decoder(x, emb)
        # x = self.last_layers(x)
        # x = self.conv_last(x)
        # x = self.decoder(x)
        return x

#----------------------------------------------------------------------------

class Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 emb_dim,
                 upscale,
                 downscale,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.upscale = upscale
        self.downscale = downscale
        init = dict(init_mode='xavier_uniform')
        self.conv_first = Conv2d(in_channels=in_channels, out_channels=emb_dim, kernel=3, **init)
        self.downsample = torch.nn.Sequential(
            torch.nn.PixelShuffle(upscale) if upscale != 1 else torch.nn.Identity(),
            torch.nn.PixelUnshuffle(downscale),
            Conv2d(in_channels=(emb_dim * downscale**2 // (upscale**2)), out_channels=out_channels, kernel=3, **init)
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.downsample(x)
        return x
  
class Decoder(torch.nn.Module):
    def __init__(self,
                 model_channels,
                 emb_channels,
                 out_channels,
                 upscale,
                 dropout,
                 swin,
                 swin_params,
                 resample_filter=[1,1]
                 ):
        super().__init__()
        self.model_channels = model_channels
        self.emb_channels = emb_channels
        self.upscale = upscale
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        init = dict(init_mode='xavier_uniform')
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, swin=swin, swin_params=swin_params
        )
        
        
        self.upsample = torch.nn.ModuleList(
            [UNetBlock(in_channels=model_channels, out_channels=model_channels, up=True, **block_kwargs),
             UNetBlock(in_channels=model_channels, out_channels=model_channels, up=True, **block_kwargs)]
        )
        self.last_layers = torch.nn.Sequential(
            GroupNorm(num_channels=model_channels, eps=1e-6),
            Conv2d(in_channels=model_channels, out_channels=out_channels, kernel=3, **init_zero)
        )

    def forward(self, x, emb):
        for layer in self.upsample:
            x = layer(x, emb)
        
        x = self.last_layers(x)
        return x


@ARCH_REGISTRY.register()
class EDMSRNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'SRNet',   # Class name of the underlying model.
        up_list         = [1, 1, 1, 1, 1],
        down_list       = [1, 2, 3, 4, 4],
        scale           = 4,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.up_list = up_list
        self.down_list = down_list
        self.scale = scale
        self.model_type = model_type
        self.model = globals()[model_type](img_resolution=img_resolution, 
                                           in_channels=img_channels, 
                                           out_channels=img_channels, 
                                           up_list=up_list[:-1],
                                           down_list=down_list[:-1], 
                                           **model_kwargs)

    def forward(self, x, x_lr, sigma, x_skip=None, class_labels=None, force_fp32=False, timestep=None, **model_kwargs):

        x = x.to(torch.float32)
        x_lr = x_lr.to(torch.float32)
        # x_cond = torch.cat([x, x_lr], dim=1).to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (((sigma / 0.1)) ** 2 + self.sigma_data ** 2)
        # c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma / 0.1) / ((sigma / 0.1) ** 2 + self.sigma_data ** 2).sqrt()
        # c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        # c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        # c_noise = (sigma).log() / 4
        c_noise = (sigma + 0.002).log() / 4
        
        ##
        # c_skip = 0
        # c_out = 1
        c_in = 1
        ##

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), lq=x_lr, class_labels=class_labels, timestep=timestep, **model_kwargs)
        # upscale = self.scale * self.up_list[timestep] / self.down_list[timestep]
        # if upscale != 1:
        #     x = torch.nn.functional.interpolate(x, scale_factor=upscale, mode='nearest')
        D_x = c_skip * x_skip + c_out * F_x.to(torch.float32)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------