from pkg import *
from utils.forward_diffusion import *

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    """
    Input: 
        - num
        - divisor
    Output:
        - arr
    e.g: num_to_groups(10, 3) = [3, 3, 3, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(
        self, 
        fn,
    ) -> None:
        super().__init__()
        """
        input:
            - fn: Neural network
        output:
            - x + fn(x)
        """
        self.fn = fn

    def forward(self, x, *args, **kwagrs):
        return self.fn(x, *args, **kwagrs) + x
    
    
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    # No more Strided Convolutions or Pooling
    return nn.Sequential(
        # Batchsize channels [optional height] [optional width] -> 
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)