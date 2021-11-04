Problem Statement
===================================

#. We have pulses.
#. Pulses consist of radial density profiles for 701 time steps
#. Pulses consist of feature profiles along the given time steps
#. We want to feed previous radial profiles and previous windows of feature profiles to predict next time step

To do this, for each pulse, we need:

#. List of ~701 radial density profiles, which for each density profile:

  #. A time window of the feature profile, with spatial distance, delta T

**Boundrary (or initial) condition**: Profile at t=0 is the initial density profile that has feature points contained in the temporal window behind it

The strohman problem would be:

Train on a pulse by pulse basis:

#. Starting from boundrary condition and feature window, predict t=1
#. Use prediction and next feature window to predict t=2
#. Use prediction from t=1, next feature window to predict t=3
#. Repeat,

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
