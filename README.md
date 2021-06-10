# TF_SWAG
Unofficial Implementation of SWAG in TensorFlow


This repository cantains the implemetation of the paper [Rethinking and Improving the Robustness of Image Style Transfer](https://arxiv.org/abs/2104.05623) in TensorFlow.

## Usage
Install the dependencies
> pip install -r requirements.txt

Run the following command
> python resnet_swag.py

Choose any one of the architecture between `resnet50_swag` and `resnet50` by using the `--architecture` flag. The default being `resnet50_swag`

>  python resnet_swag.py --architecture='resnet50'

Change the number of iterations by using the `--iter` flag
>  python resnet_swag.py --iter=500

## Acknowledgement
The code is heavily borrowed from the [original implementation](https://github.com/peiwang062/swag) of the paper
