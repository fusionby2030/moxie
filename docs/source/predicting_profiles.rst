Generating Profile(s)
===================================

We want to use Variational Autoencoders to generate density and temperature profiles in the edge/scrape-of-layer.

This is a semi-unsupervised learning approach, where we use the profiles to then predict profiles, and in doing so we encode the information in a latent space, that can be used to generate new samples.

Goals
~~~~~

#. Generate density (and tempearture) profiles using VAEs
#. Encode machine control parameters into the inputs and or latent space (dreaming)

  * Dual headed VAE, with one highway taking inputs, the other taking profiles, then concat before latent space.
  * Entanglement of latent space with inputs? Not sure how this works but sounds fancy
  * Two separate VAEs with entangled latent space, then modular switch encoding components to get input -> profile
#. Extent machine control parameters + profile inputs into a time evolving predictor
  * Take previous time step params + profiles to predict next time step profile
#. Diagnostics as inputs???

Accomplished
~~~~~

* Vanilla VAE's
* Vanilla Convoluitonal VAE's

TODO's
~~~~~

1. Rework ConvVAE to be able to take both Te and Ne profiles

  * Rework Experiment class to be able to plot Te from given Ne for test set

2. Add Model Documentation
  * Vanilla FF VAE
  * ConvVAE

3. Add General Documentation, which will require going through pytorch lightning
  * Logger?


Models
-------

All models are found in the :file:`src/models/` and are written with pytorch.

1. Vanilla VAE

  * Simple fully connected linear layer model
  * TBD: Activation function
  * TBD: KL-Div weighting hyperparam for loss function, as it needs to be quite small or else the recon loss dominates and the model just spits out the (literal) average profile found in the training set

2. Convolutional VAE

  * TBD: Variable stride


Experiments
-------

We use pytorch lightning, but this is subject to change.
See :file:`src/experiment.py` and :file:`src/run.py`
