Generating Time-Independent Profile(s) [PSI 2022]
=================================================

We want to use Variational Autoencoders to generate density and temperature profiles in the edge/scrape-of-layer.

This is a semi-unsupervised learning approach, where we use the profiles to then predict profiles, and in doing so we encode the information in a latent space, that can be used to generate new samples.

Goals
~~~~~

1. Generate density (and tempearture) profiles using VAEs
2. Encode machine control parameters into the inputs and or latent space (dreaming)

  * Dual headed VAE, with one highway taking inputs, the other taking profiles, then concat before latent space.
  * Entanglement of latent space with inputs? Not sure how this works but sounds fancy
  * Two separate VAEs with entangled latent space, then modular switch encoding components to get input -> profile

3. Extent machine control parameters + profile inputs into a time evolving predictor

  * Take previous time step params + profiles to predict next time step profile
4. Diagnostics as inputs???
5. Establish physics informed neural networks (PINN)

   * Requiring plasma edge to have 0 temperature and density (in the loss?)

Accomplished
"""""""""""""

* A simple 1D Convolutional VAE that can represent profiles given the previous profiles.
* Initial Latent space discovery shows that even 4 dim can recreate the profile
* Beta + Gamma VAEs coded, but not optimized

Initial Results
""""""""""""""""

.. image:: ./images/Singular_latent_space_aaro.png
  :width: 400

.. image:: ./images/Singular_profile_space_aaro.png
  :width: 400




TODO's
~~~~~~~~~~

1. Model development

  * Allow input of Te and Ne profiles into model
  * Layers and how to stack them

2. Visualizations

  * Output Plots
    * Latent Space
    * Profiles
  * Clustering of predictions from latent space (see how the model is actually grouping profiles)
  * Layer by layer output of conv. and tranposed. blocks to see what features the model is deeming important

3. Experiments

  * Rework Experiment class to be able to plot Te from given Ne for test set
  * Achieve a successful Beta run

4. Data

  * Psep
  * Diagnostics?

Models
-------

All models are found in the :file:`src/models/` and are written with pytorch, see the models page for more info.


Experiments
--------------

We use pytorch lightning, but this is subject to change.
See :file:`src/experiment.py` and :file:`src/run.py`
