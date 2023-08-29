import torch
import torch.nn as nn
from utils.utils import *
from utils.bottleneck import ResnetBlock
from utils.pos_embed import SinusoidalPositionEmbeddings
from utils.group_norm import PreNorm
from utils.attention import Attention, LinearAttention


class Unet(nn.Module):
    def __init__(
        self,
        dim,                    
        init_dim=None,          # if init_dim is None: init_dim = dim
        out_dim=None,           # if out_dim is None: out_dim = channels
        dim_mults=(1, 2, 4, 8), # define number layers of downsample and upsample
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # first, a convolutional layer is applied on the batch of noisy images
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        # if dim = 28, init_dim = 28 => dims = [28, 28, 56, 112, 224]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # if dim = 28, init_dim = 28 => in_out = [(28, 28), (28, 56), (56, 112), (112, 224)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Module to add time embedding
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings, position embeddings are computed for the noise levels
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # init layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Second, each downsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + a downsample
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            # a sequence of downsampling stages are applied.
            self.downs.append(
                nn.ModuleList(
                    [
                        # block1
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        # block2
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        # Attention
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        # Downsample
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # Third, at the bottleneck, again ResNet blocks are applied, interleaved with attention
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Fourth, each upsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + a upsample
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            # a sequence of upsampling stages are applied.
            self.ups.append(
                nn.ModuleList(
                    [
                        # block1
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        # block2
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        # Attention
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        # Upsample
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        # finally, a ResNet block followed by a convolutional layer is applied
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # h to store value for skip connection
        h = []

        # Downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)