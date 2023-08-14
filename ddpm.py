from typing import Tuple, Optional          # strong-typing support

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from utils import gather


class DenoiseDiffusion:
    """
    Denoise diffusion
    """
    
    def __init__(
        self,
        eps_model: nn.Module,
        n_steps: int,
        device: torch.device,
    ) -> None:
        """
        * `eps_model` is $\epsilon_\theta(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        
        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        
        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # $T$
        self.n_steps = n_steps
        
        # $\sigma^2 = \beta$ is variance
        self.sigma2 = self.beta
        
    def q_xt_x0(
        self, 
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get $q(x_t|x_0)$
        """
        pass