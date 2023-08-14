# Denoising Diffusion Probabilistic Models (DDPM)

In simple terms, we get an image from data and add noise step by step. Then we train a model to predict that noise at each step and use the model to generate images

Here is the Unet model that predicts the noise and traing code. This file can generate samples and interpolations from a trained model