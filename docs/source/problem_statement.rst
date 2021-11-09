Problem Statements
===================================

There are various problems we can solve using the large number of pulses contained. 
The first is the density at the separatrix, then general profile predictions. 

Separatrix Density Prediction
------

JET pedestal database method (PSI Abstract)
~~~~~

The JET ped database contains time averaged (between t1 and t2) pulses with corresponding values for nesep. 
The values are calculated using 2-point method, where T(Rsep) = 100eV is assumed and the profiles are shifted to have nesep at the corresponding Rsep. Using these pulses, we can extract the profiles that were averaged (between t1 and t2) and assign the value of nesep to them, building a supervised learning dataset, with input -> ne profile, output -> nesep. To ensure not complete overfitting in the 300 values of nesep. Some variations can be done to make the experiment more robust. 

#. The profiles that range for t1 to t2 can be assigned different values of nesep, which could be sampled from a gaussian distribution with mean of the time averaged (database value) of nesep, and width of standard deviation of time averaged value of nesep (database value of error). 
#. We can feed both the ne and te profiles to determine nesep


To do this, we will need to construct the supervised learning dataset. 

* |check| Gather pulses from JET pedestal database (ne, te profiles for each pulse) 
* |check| Cut pulse profiles between the t1, t2 for corresponding entry in JET PDB
* |check| Make list of X, y, with X list of neped profiles, y list of corresponding neseps 

This is stored in the raw folder! 

Now we will feed this into a model, to be determined later...

Using the inputs Method
~~~~~


* We have pulses 
* Pulses consist of radial density and temperature profiles for 701 time steps
* Also consist of feature profiles along the given time steps 
* We want to feed previous feature profiles to predict the nesep in a time evolving maner 


To do this, for each pulse, we need: 

* nesep at the 701 time steps 
        * Find datapoints around Te = 100eV
        * The mean radius of those points is then the position of separatrix 
        * get ne at that position by averaging the points around the separatrix 
        * ????? 
        * Profit 
* For each time step, window the inputs to feed to an RNN. 


To check that we get the correct nesep values, we can check with the JET pedestal database for time windows given. 


Density Profile Prediction
-----

#. We have pulses.
#. Pulses consist of radial density profiles for 701 time steps
#. Pulses consist of feature profiles along the given time steps
#. We want to feed previous radial profiles and previous windows of feature profiles to predict next time step

To do this, for each pulse, we need:

#. List of ~701 radial density profiles, which for each density profile:

  #. A time window of the feature profile, with spatial distance, delta T

**Boundrary (or initial) condition**: Profile at t=0 is the initial density profile that has feature points contained in the temporal window behind it

Srohman Method 
~~~~~

Train on a pulse by pulse basis:

#. Starting from boundrary condition and feature window, predict t=1
#. Use prediction and next feature window to predict t=2
#. Use prediction from t=1, next feature window to predict t=3
#. Repeat,


Extended Strohman method 
~~~~~

A variation would be to either vary:

#. Set a parameter N, which is number of previous windows or prediction to feed and aid in the next prediction
  #. N amount of previous feature windows are given to network to predict next profile
  #. N amount of previous predictions are given to network to predict next profile


For including both and N=3, the training would look like:

#. Starting from boundrary condition and feature window, predict t=1
#. Use prediction, and previous + next feature window to predict t=2
#. Use prediction from t=1, t=2, and all previous feature windows + next feature window to predict t=3
#. Use prediction from t=1, 2, 3, and all previous feature windows + next feature windows to predict t=4
#. Use prediction from t=2, 3, 4 and feature windows from t=2,3,4 + next feature window to predict t=5
#. Repeat,



.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">
