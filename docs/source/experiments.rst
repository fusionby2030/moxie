Experiments
===========


- How do we know that the VAE is *working* and what does *working* mean? 

To this end, we have a few requirements that would suggest that the VAE is learning enough to be considered a *working* model. Each requirement is built upon the previous one, so at the end they all co-exist. 

1. A decent profile reconstruction 
  
  - For starters, we want the profiles reconstructed to have a a relatively low RMSE (a few percentage points) on average when simply encoding and decoding the profiles.
  - QUANTIFICATION: RMSE 

2. A decent machine parameter reconstruction 

  - Same as above, but for machine parameter reconstructions. This will help also determine with which machine parameters the model struggles with predicting and which ones are well predicted. NOTE:  machine parameters are also being predicted from :math:`Z_{mach}`.  
  - QUANTIFICATION: RMSE, Global and Individual (per machine parameter)

3. :math:`Z_{stoch}` and :math:`Z_{mach}` provide reasonable compressions of the plasma state 

  - We would like that when we assume a general set of control parameters, i.e., fix :math:`Z_{mach}` the resulting decoded profiles will vary *stochastically* proportional to the variations found in experimental data. An example of this is when the profile shifts up and down due to ELMs, thus over the course of multiple time slices, the experimental profiles shift 'up/down' (relative to density/temeprature) due to ELMs spitting out energy to the SOL. Therefore, by sampling from :math:`Z_{stoch}`, we should see these relative shifts, as it would not necessarily have to do with the machine parameters. 

  - :math:`Z_{mach}` should give a good guess of the mean of the pulses with similar parameters (like above). 

4. A decent understanding of physics 

  - Sweeping the dimensions that have to do with :math:`I_P` should show linear correlations in density

  - :math:`\alpha^*`: TBD
