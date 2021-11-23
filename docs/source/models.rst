Model(s)
===================================

We want to use unsupervised representation learning to reproduce and generate profiles.
From the generated profiles, we could downstream use supervised learning to determine nesep for example.


.. autoclass:: moxie.models.VisualizeBetaVAE
  :members:
  :inherited-members:
  :exclude-members: 

VAE(s)
--------------

There are a few parameters important to vary in the VAE:

* Latent space dimensions
* Deepness of network
* Regularization of KL divergence term in loss

**Loss term**

.. math::

   Loss = Exp_{q(z|x)}[\log (recon) - D_{KL}]

Disentaglement
~~~~~~~~~

**Independent(!!) features of your input are ideally encoded into the latent space of the model.**

Examples:

* Representing photos of people
  * Clothing is independent of height, wheras length of left leg is dependent on length of right leg
  * Disentangled representation would encode height and clothing into *separate* dimensions of the latent space.

If we have a learned latent space that independently encodes a persons height, then we can modify that while keeping everything else the same and generate a new sample.

A way of doing this is with a :math:`\beta`-VAE [betavae]_ which involves simply adding a hyperparameter, :math:`\beta` to the KL-divergence term in the loss.
The KL-div term quantifies how different the encoded distribution of the latent space is from a prior, i.e., penalizing the network when the latent space distributions are different from e.g. Gaussian prior.
Since the network encods the prior with a mean and variance, in the example of a gaussian prior, the mean will be pushed ot 0, and variance to 1 and corrrleation between the dimensions is 0.
The :math:`\beta` term further constrains the mean values in terms of how much they can differ between different observations.

If we crank up :math:`\beta`, the latent vector changes in the following way:

1. More smooth changes in variations of latent vector
2. Compress more information about the dataset into as few dimensions of z as possible
3. Align main axis of the training data variability with dimensions of the latent space


**THERE IS A PROBLEM**

My implementation only likes small betas, so we will try to give a small beta to start, then slowly increase it overtime.



.. [betavae] Beta-vae citation
