# Denoising Diffusion Probabilistic Models

## What is a diffusion models?

- Is a neural network learns to gradually denoise data starting from pure noise
- 2 processes, *T* timesteps:
    - a fixed forward diffusion process $q$
    - a learned reverse denoising diffusion process $p_\theta$

## In more mathematical form

- $q(x_0)$ is the real data distribution. Intuition, we utilize denoising diffusion process to approximate the real data distribution
- $q(x_t | x_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I), \text{which } \beta_t \text{ is a known variance schedule}$
- if we know the conditional distribution $p(x_{t-1}|x_t)$, then we could denoise and end up with the real distribution $q(x_0)$
- Learnable parameter:
    - a mean parametrized by $\mu_\theta$
    - a variance parametrized by $\Sigma_\theta$, for easier to train, the authors keep the variance fixed

## Defining an objective function (by reparametrizing the mean)

- Combination of $q$ and $p_\theta$ can be seen as a VAE
- The **variational lower bound** (ELBO) can be used to minimize the negative log-likelihood with respect to $x_0$
- ELBO for this process is a sum of losses at each time step t $L = L_0 + L_1 + ... + L_T$
- Each term $L$ is actually the **KL divergence between 2 Gaussian distributions** 

## The neural network

- U-Net, consists of "bottleneck" to ensure the network learns only the most important information

## Network helpers

## Position embeddings (t timestep)

- Encode timestep t to make the neural network "know" at which particular time step (noise level) it is operating, for every image in a batch.

## ResNet block

- U-Net Encoder

## Attention Module

## Group Normalization

## Conditional U-Net

- Define the entire neural network, the job of $\epsilon_\theta(x_t, t)$ is to take in a batch of noisy images and their respective noise levels, and output the noise added to the input
- The network is built up as follows:
    - first, a convolutional layer is applied on the **batch** of noisy images, and position embeddings are computed for the noise levels
    - next, a sequence of **downsampling** stages are applied. Each downsampling stage consists of *2 ResNet blocks* + *groupnorm* + *attention* + *residual connection* + *a downsample*
    - at the **bottleneck**, again ResNet blocks are applied, interleaved with attention
    - next, a sequence of **upsampling** stages are applied. Each upsampling stage consists of *2 ResNet blocks* + *groupnorm* + *attention* + *residual connection* + *a upsample*
    - finally, a ResNet block followed by a convolutional layer is applied
