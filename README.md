# DIP-VAE

An implementation of the DIP-VAE algorithm from the paper

Variational Inference of Disentangled Latent Concepts from Unlabelled Observations (Kumar et al, ICLR 2018)
https://arxiv.org/abs/1711.00848

You will need to download the file dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz from https://github.com/deepmind/dsprites-dataset and place it in the root of this repo.

To train the model, run

python dip_vae.py [i|ii] train

The argument i or ii specifies whether to train DIP-VAE-I or DIP-VAE-II 

Checkpoints will be saved in ./[i|ii]_checkpoints
The latest checkpoint can be loaded (for e.g. an interactive session) by running

python -i dip_vae.py [i|ii] load
