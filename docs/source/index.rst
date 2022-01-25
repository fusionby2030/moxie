Welcome to Moxie's documentation!
===================================

**Moxie** is a Python tool to analyze pedestal profiles in MCF devices.


.. note::

   This project is under active development.

Long Term Plan
---------------

* Live pedestal/SOL predictions for EUROfusion devices using machine control parameters (C) and available diagnostics (D)

.. image:: ./images/end_goal.svg.png
  :width: 200
  :alt: Version 1 of Long term model

1. Generate density (and tempearture) profiles using VAEs
2. Encode machine control parameters into the inputs and or latent space (dreaming)

  * VAE with two latent spaces, `z_stochastic, z_machine`

3. Extent machine control parameters + profile inputs into a time evolving predictor

  * Take previous time step params + profiles to predict next time step profile
4. Diagnostics as inputs???
5. Establish physics informed neural networks (PINN)

   * Requiring plasma edge to have 0 temperature and density (in the loss?)


This 'meta-model' is comprised of many **modular** sub-models, which can operate independently.
The current list of sub-models:

* VAE for single timestep predictions

  * Determining relevant SOL parameters like :math:`n_{e, sep}, P_{sep}` from latent space or profile of single time step

Contents
--------

.. toctree::

   predicting_profiles
   models
   literature_review
   data
   datasets
   nesep_problem_statement
