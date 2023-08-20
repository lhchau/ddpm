#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Denoising Diffusion Probabilistic Models",
    author="Hoang-Chau Luong",
    author_email="lhchau20@apcs.fitus.edu.vn",
    url="https://github.com/lhchau/ddpm",
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
)
