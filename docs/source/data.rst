Data Storage
===================================

This subdirectory contains the various forms of datasets used in the analysis.


| data
| ├── raw_datasets < original immutable data dump from HEIMDALL
| |  ├── all_shots.pickle
| ├── processed  < final, canonical datasets for modeling
| |  ├── pedestal_profile_dataset_v3.pickle


Processed
----------

Dataset(s)
""""""""""""""

The code for this section is found in :file:`/src/data/`

1. JET Pedestal Database (JPDB)

  * We use the entries found in the established DB between the time averaged windows given.
  * These are all flat top H-mode entries
  * Currently called v3 in sceibo

2. Extension of JPDB

  * Use time windows outside of those found in the JET DSP
  * Still use the same pulses found in the DB, but this will include L-mode, as well as L-H mode transition profiles

3. All HRTS validated shots >= 79000

  * Yeah. Big data energy.

Description of Datasets
""""""""""""""""""""""""""""

We will take temperature and density profiles from HRTS scans, as well as the machine control parameters for the entire duration of the pulse. Additionally, we can grab any and all diagnostic equipment we may like.

1. We initially grabbed all HRTS validated shots with shot number >= 79000.

  * These are stored in dictionary format in a pickle file. If you have the file, then each key in the dictionary is a pulse number
  * Each pulse is another dicitonary with keys: `'inputs', 'outputs'`
  * Inputs is a dictionary, with keys corresponding to the control parameters
    * Each control parameters is a dictionary, with keys `'values', 'time'`
  * Outputs is a dictionary with keys `'NE', 'DNE', 'DTE', 'TE', 'radius', 'time'`
  * If you know you know

2. 82557 total profiles from 2176 HRTS validated pulses found in JPDB (see :file:`/src/data/create_psi_database.ipynb`)

  * These are then stored in an HD5Y file
  * The (current, V3) HD5Y file is organized into two data groups: `'strohman' and 'density_and_temperature'`
  * There is additionally the `'meta'` group, which has `'pulse_list', 'y_column_names'` which store arrays regarding which pulses and what the columns of the y vector relate to.
  * These two groups are structured with three subgroups: `'train', 'valid', 'test'`
  * Each subgroup has 2 datasets:   `'X', 'y'`, where `'X'` has the inputs (profiles) and `'y'` has the machine parameters and nesep

Example of accessing the 2 channel density and temperature profile looks like this:

.. code-block:: python

  with h5py.File('../processed/pedestal_profile_dataset_v3.hdf5', 'r') as file:
  group = file['density_and_temperature']
  X_train, y_train = group['train']['X'][:], group['train']['y'][:]
  X_test, y_test = ...


Data-splitting
""""""""""""""

For each pulse, we should take 70% of the profiles for training, 10% for validation, and 20% for testing. This will ensure that each pulse is represented in each dataset.

See above

Preprocessing and DataClasses
""""""""""""""

Currently, we just take the max density value for the training set and divide all ne points by that value. This constrains the input profiles to be between 0 and 1. This is subject to change.
The dataclasses are stored in :file:`src/data/profile_dataset.py`


Raw Datasets
----------

**Total HRTS validated shot count: 4942 Shots.**

Stored in the following raw format:

.. code-block:: python

  {'79100': {'inputs': {'BT': {'values': np.array, 'time': np.array},
                          'IP': {'values': np.array, 'time': np.array}, ...}
              'outputs': {'time': np.array, 'radius': np.array, 'NE': 2D np.array, 'DNE': 2D np.array }
              },
  '79103': {'inputs': {'BT': {'values': np.array, 'time': np.array},
                          'IP': {'values': np.array, 'time': np.array}, ...}
              'outputs': {'time': np.array, 'radius': np.array, 'NE': 2D np.array, 'DNE': 2d np.array}
              },
              }


Input Columns
~~~~~~~~~~~~

All are numpy arrays with shape given. The keys in the raw inputs dictionary are written below.

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



Output Profiles
~~~~~~~~~~~~

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
