#!/bin/bash

# SMIE
# dataset ufo; learning rate 0.001; epoch 200; rank 5; gpu 0
python test_SMIE.py -d ufo -l 0.001 -e 200 -r 5 -c 0

# SNETD
# dataset ufo; learning rate 0.001; epoch 200; rank 5; window size 50; gpu 0
python test_SNETD.py -d ufo -l 0.001 -e 200 -r 5 -w 50 -c 0
