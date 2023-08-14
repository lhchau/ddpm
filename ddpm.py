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
        Forward:
        Get $q(x_t|x_0)$ distribution
        
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        """
        
        # [gather] $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t) 
        
        return mean, var
    
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        Sample from $q(x_t|x_0)$
        
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        """
        
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.rand_like(x0)
        
        # get $q(x_t|x_0)$ distribution
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps
    
    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor
    ):
        """
        Sample from p_\theta(x_{t-1}|x_t) = \mathcal{N}\big(x_{t-1};
                    \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I} \big)
                    \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        """
        # \epsilon_\theta(x_t, t)
        eps_theta = self.eps_model(xt, t)
        # [gather] $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t})$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)
    
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps
    
    def loss(
        self,
        x0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # get batch_size 
        batch_size = x0.shape[0]
        # get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.rand_like(x0)
            
        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get \epsilon_\theta(\sqrt{\bar\alpha_t} + x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)