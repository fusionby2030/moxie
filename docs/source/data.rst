Data Storage
==============

This subdirectory contains an overview of the profile database used in the analysis.

.. code-block:: text

  data                                        - the data dir, (if have it know you know)
  ├── raw_datasets                            - raw data dumps
  │   ├── all_shots.pickle                    - original data dump profile from HEIMDALL
  ├── processed                               - final, canonical datasets for modeling
  │   ├── pedestal_profile_dataset_v3.pickle  - The first version of the pedestal database, has missing input values, do not use
  │   ├── profile_database.hd5y               - The current version of the pedestal database, description below

To see more information on specific datasets, see :doc:`/datasets`

Profile Database
----------------

The profile database is an hd5y file which houses many pulses and the relevant machine parameter and profile data. If you don't know how to use hd5y files, then check out `HD5Y <https://docs.h5py.org/en/stable/index.html>`_
Below is an example to get the toroidal field and time steps as well as the density profile data, radius and time steps of pulse 86932:

.. code-block:: python

  with h5py.File('profile_database.hdf5', 'r') as file:
    bt_vals = file['86932/machine_parameters/BT/values'][:] # the [:] at the yields the dataset in numpy array form
    bt_time = file['86932/machine_parameters/BT/time'][:]

    de_data = file['86932/profiles/NE'][:]
    de_radius = file['86932/profiles/radius'][:]
    de_time = file['86932/profiles/time'][:]


Pulse Structure
""""""""""""""""
The pulses in the database are structured in the following format:

* Each pulse is a group

  * For each pulse, there are the following groups

    * :code:`'machine_parameters'`, which contain relevant machine parameters as subgroups following subgroups, within each there are datasets named:

      * :code:`'time'`: The time steps at which the machine parameter is sampled during a pulse
      * :code:`'values'`: The value of the machine parameter for each given time step in :code:`'time'`
      * Each of the above datasets are the same length (naturally)

  * To check which machine parameters are stored, run :code:`file['pulse_id/machine_parameters'].keys()`

  * Within each profile group there are the following datasets

    * :code:`'radius'`: the spatial resolution of the HRTS scan (this is R (m) and **NOT** the normalized flux coordinate)
    * :code:`'time'`: The temporal resolution of the HRTS scan
    * :code:`'NE'`: The density profiles, a 2D vector, with rows corresponding to each time step given in :code:`'time'`, and columns corresponding to the values in :code:`'radius'`, with each entry being the electron density
    * :code:`'DNE'`: The error in the density profiles, similar structure to above
    * :code:`'TE'`: The temperature profiles, similar structure to above
    * :code:`'DTE'`: Error in temperature profiles, similar structure to above


If one wanted to update or add information to the pulse, then simply write to the file using the above syntax!
For example, to add a machine parameter or toroidal flux coordinate for the profiles you could do the following:

.. code-block:: python

  with h5py.File('profile_database.hdf5', 'r+') as file:
    my_new_vals = np.array([1., 1., 1.])
    my_new_time_array = np.array([1., 1., 1.])
    new_vals = file['86932/machine_parameters/new_parameter'].create_dataset('values', my_new_vals)
    new_time = file['86932/machine_parameters/new_parameter'].create_dataset('time', my_new_time_array)

    my_flux_coords = np.array([0, 0.5, 1])
    flux_radius = file['86932/profiles'].create_dataset('flux_coords', my_flux_coords)



Machine Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The current machine parameters stored for each pulse in :code:`'machine_parameters'`. The top-level is the dda, whereas the sub-lists are the actual names of the parameters.

* EFIT: shape (989,)
	* Q95
	* RGEO (Major R)
	* CRO (Minor radius = (Rmax-Rmin)/2 )
	* VOLM (plasma volume)
	* TRIU and TRIL (Upper and lower triangularity)
	* XPFL (limiter and divertor flag)
	* XIP (Plasma current measured)
	* ELON (Elongation)
	* POHM (Ohmic Power)

The time resolution for each EFIT parameter is given as EFIT_T in the inputs dictionary.

Each of the next columns have both the value and time stored in it as a dicitionary with keys 'values', 'time'

* SCAL: shape (1024,)
	* BT (Toroidal Field)
* GASH: shape (8101,)
	* ELER (electron flow rate)
* NBI: shape (13104,)
	* PTOT (Total Neutral Beam Power)
* ICRH: shape (1000,)
	* PTOT (Total ICRH power)


Profiles
~~~~~~~~~~~~~~~~~~~~~~~~

The profile data from each pulse is stored in the :code:`'profiles'` subgroup.

* Density (NE)
  * 2D array: shape (701, 63)
* Error (DNE)
  * 2D array: shape (701, 63)
* Density (TE)
  * 2D array: shape (701, 63)
* Error (DNE)
  * 2D array: shape (701, 63)
* Temporal (time)
  * Temporal resolution of profile, shape (701,) corresponds to the rows of above profile array
  * This changes depending on the pulse
* Radial (radius)
  * Spatial resolution of profile
  * This changes depending on the pulse

Shape is (701, 63) for each pulse, where 701 and 63 are the temporal and spatial resolution respectively.
