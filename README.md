# Fitting PyTorch models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

<img src="icon.png" align="center" width="400px"/>

This repository follows a workshop to set up a Python package, build some neural networks with `torch`, and publish the models on HuggingFace. 

There are two datasets that we simulate:
1. Colored shapes (circle, rectangle, triangle) in a pixellated image
2. One-dimensional sinusoids

Alongside these datasets, we fit the following model objectives:
1. (Classification) Predict the shape and color in the image
2. (Regression) Predict the next time steps of the sine function
