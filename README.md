# Research Collection

This repository is a collection of various projects in the area of machine learning and data science. *Most* of these projects are written in a Jupyter Notebook for clear visibility and reproducibility. Notebooks ( `.ipynb` files ) can natively be viewed in GitHub, but lack some functionality, so it is recommended to use [jupyter nbviewer](https://nbviewer.jupyter.org/github/stockeh/research/tree/master/) to read.

## Generative Adversarial Networks

Exploratory project focused on providing insight to the structure, training process and evaluation of Generative Adversarial Networks ( GANs ). MNIST data is used to display and contrast the findings of different models performance.

A modified Inception Score is implemented to evaluate how well generated images of numbers from a vanilla GAN compares to those using a deep convolutional GAN. Synthetic results are displayed and mathematically justified throughout the notebook.

[ [gans](https://github.com/stockeh/research/tree/master/gans) | [notebook](https://nbviewer.jupyter.org/github/stockeh/research/blob/master/gans/gan-project.ipynb) ]

## Observing Celestial Objects

Research oriented project exploring imagery and numerical spectra data from the Sloan Digital Sky Survey. This utilized neural networks and statistical approaches to classify and better understand stars, galaxies, and quasars.

Algorithms such as Scaled Conjugate Gradient ( SCG ), and those provided by PyTorch; Adaptive Moment Estimation ( Adam ), and Stochastic Gradient Decent with momentum ( SGD ), were experimented with to accurately represent the data with upwards of 98% **testing** accuracy.

[ [star-search](https://github.com/stockeh/research/tree/master/star-search) | [notebook](https://nbviewer.jupyter.org/github/stockeh/research/blob/master/star-search/stock-starsearch.ipynb) ]
