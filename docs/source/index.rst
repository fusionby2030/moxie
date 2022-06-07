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


Roadmap
-------

Each point is successive, i.e., built upon the previous. 

1. Working VAE for time-independent pedestal profiles in JET

  * Latent spaces encode relevant information, 
    
    * :math:`Z_{mach}` should give a good guess if not the mean of all the time slices with similar machine parameters 
    * Varying :math:`Z_{stoch}` from above :math:`Z_{mach}` samples should cover variation in relative shifts (e.g., vertical) in time slices
   
  * Physics checks out, (examples below)
    
    * Sweeping :math:`I_P` should linearlly increase density
    * Pulses with pellets should push :math:`n_{e, ped}` up relative to SOL. 
    * predicted :math:`\alpha^*` (and other physical parameters) follow general understanding of SOL/pedestal physics (see EPED)
  
  * Encode physics 

    * ELM timing as input 
    * Encode the experimental physics studies above into learning process.   

  * Decode physics 
    
    * Predict :math:`n_{e, ped}, P_{SOL}`

  * Ablation studies of varying levels of complexity in dataset, i.e., dataset with pellets vs without pellets. 

2. Establish multi-machine time-independent VAE with gathered AUG data 

  *  Transfer learning

3. Potential next steps

  * Time-dependent
  * Grid invariant (regardless of # of measurements or points, the output is encoded to machine normalized quantities)

Contents
--------

.. toctree::

   experiments
   predicting_profiles
   literature_review
   data
   datasets
