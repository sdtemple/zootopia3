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


.
├── LICENSE
├── README.md
├── answers
│   ├── modeling-colored-shapes.ipynb
│   ├── modeling-waves.ipynb
│   └── simulate.ipynb
├── data
├── docs
├── examples
├── how-train-nns-sethtem.pdf
├── icon.png
├── pyproject.toml
├── src
│   ├── zootopia3
│   │   ├── __init__.py
│   │   ├── circle.py
│   │   ├── diamond.py
│   │   ├── experimental
│   │   │   └── __init__.py
│   │   ├── figures
│   │   │   └── __init__.py
│   │   ├── image.py
│   │   ├── models
│   │   ├── models.py
│   │   ├── rectangle.py
│   │   ├── shapes
│   │   │   ├── __init__.py
│   │   ├── simulate.py
│   │   ├── sine.py
│   │   ├── triangle.py
│   │   ├── utils
│   │   │   └── __init__.py
│   │   └── waves
│   │       ├── __init__.py
│   └── zootopia3.egg-info
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       ├── requires.txt
│       └── top_level.txt
└── tests
