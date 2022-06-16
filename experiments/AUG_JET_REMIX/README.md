# Meditations on getting AUG vs JET going

## Goal

Representation learning for JET and AUG data

## Steps

1. First train on AUG using torch/pytorch lightning

## Structure/libraries:

### Model 

- torch (torch<=1.11) 
- Simple VAE with encoder and decoder
- Might have to figure out the transfer of dimensionality for higher resolution of IDA...

### Data: 
    
- Make use of pythons dataclasses to parse raw data 
    - `create_datasets.py`
- Feed to torch dataloaders