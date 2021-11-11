Generating Profile(s)
===================================


We want to use Variational Autoencoders to generate density and temperature profiles in the edge/scrape-of-layer.

This is a semi-unsupervised learning approach, where we use the profiles to then predict profiles, and in doing so we encode the information in a latent space, that can be used to generate new samples.

Goals
~~~~~

#. Generate density (and tempearture) profiles using vanilla VAEs
#. Encode machine control parameters into the inputs and or latent space (dreaming)
  * Dual headed VAE, with one highway taking inputs, the other taking profiles, then concat before latent space.
  * Entanglement of latent space with inputs? Not sure how this works but sounds fancy
  * Two separate VAEs with entangled latent space, then modular switch encoding components to get input -> profile
#. Extent machine control parameters + profile inputs into a time evolving predictor
  * Take previous time step params + profiles to predict next time step profile
#. Diagnostics as inputs???

Accomplished
~~~~~

* Vanilla VAE's that can generate profiles okay.

Dataset(s)
---------

#. JET Pedestal Database (JPDB)
  * We use the entries found in the established DB between the time averaged windows given.
  * These are all flat top H-mode entries
#. Extension of JPDB
  * Use time windows outside of those found in the JET DSP
  * Still use the same pulses found in the DB, but this will include L-mode, as well as L-H mode transition profiles
#. All HRTS validated shots >= 79000
  * Yeah. Big data energy.

Description of Datasets
~~~~~

We will take temperature and density profiles from HRTS scans, as well as the machine control parameters for the entire duration of the pulse. Additionally, we can grab any and all diagnostic equipment we may like.

#. Total number of entries?
#. List of control parameters?


Data-splitting
~~~~~

For each pulse, we should take 70% of the profiles for training, 10% for validation, and 20% for testing. This will ensure that each pulse is represented in each dataset.

Preprocessing and DataClasses
~~~~~

Currently, we just take the max density value for the training set and divide all ne points by that value. This is subject to change.
The dataclasses are stored in `src/data/profile_dataset.py`



Models
-------

All models are found in the `src/models` directory, and are pytorch.
#. Vanilla VAE
  * Simple fully connected linear layer model
  * TBD: Activation function
  * TBD: KL-Div weighting hyperparam for loss function, as it needs to be quite small or else the recon loss dominates and the model just spits out the (literal) average profile found in the training set
#. Convolutional VAE
  * TBD: Everything!


Experiments
-------

We use pytorch lightning, but this is subject to change.
See `src/experiment.py` and `src/run.py`
