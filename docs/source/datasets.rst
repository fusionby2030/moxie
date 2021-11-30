Dataset(s) in Profile Database
=================================

Within the profile database, there are various datasets that use the pulse information stored above. They are found in the top-level subgroup :code:`'processed_datasets'`.

.. code-block:: text

  profile_database           - the database file, (if have it, you know)
  ├── 87500                  - Shot # as explained in data
  ├── 87501                  - Shot # as explained in data
  ...
  ├── processed_datasets     - final, canonical datasets for modeling
  │   ├── PSI22              - The current version of the PSI dataset


Instructions of how to make your own dataset are TBD, but for the moment you can check :file:`/src/data/create_psi_database.ipynb`

The pseudocode goes like this:

.. code-block:: python

  with h5py.File('../processed/pedestal_profile_dataset_v3.hdf5', 'r') as file:
    new_dataset_group = file['processed_datasets'].create_group('new_dataset')
    # take data from original pulses and store them in this new dataset,


PSI22
----------
The first dataset is for PSI 2022, i.e., it is found under :code:`/processed_datasets/PSI22`.


1. 82557 total profiles from 2176 HRTS validated pulses found in JPDB (see :file:`/src/data/create_psi_database.ipynb`)

  * These are
  * The (current, V3) HD5Y file is organized into two data groups: `'density' and 'density_and_temperature'`
  * There is additionally the `'meta'` group, which has `'pulse_list', 'y_column_names'` which store arrays regarding which pulses and what the columns of the y vector relate to.
  * The `'density' and 'density_and_temperature'` subgroups are populated with three subgroups: `'train', 'valid', 'test'`
  * Each subgroup has the follwoing datasets:
    * `'X'`: has the inputs (profiles)
    * `'y'`: has the machine parameters and nesep
    * `'radii'`: has the toroidal radius for each profile



Example of accessing the 2 channel density and temperature profile looks like this:

.. code-block:: python

  with h5py.File('../processed/pedestal_profile_dataset_v3.hdf5', 'r') as file:
    group = file['density_and_temperature']
    X_train, y_train, radii_train = group['train']['X'][:], group['train']['y'][:], group['train']['radii'][:]
    X_test, y_test, radii_test = ...



Detailed Description of PSI22 database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What shots are included in PSI22?
""""""""""""""""""""""""""""""""""""""""""

1. JET Pedestal Database version (JPDB)

  * We use the entries found in the established DB between the time averaged windows given.
  * These are all flat top H-mode entries
  * Currently called v3 in sceibo
  * TBD, it will be included in the profile database under the group :code:`'JETPDB_1'`

2. Extension of JPDB, **Not implemented yet**

  * Use time windows outside of those found in the JET PDB
  * Still use the same pulses found in the DB, but this will include L-mode, as well as L-H mode transition profiles

3. All HRTS validated shots >= 79000 **Not implemented yet**

  * Yeah. Big data energy.

What data from each shots are included in PSI22?
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

We take temperature and density profiles from HRTS scans, as well as the machine control parameters for the entire duration of the pulse. Additionally, we can grab any and all diagnostic equipment we may like.

1. We initially grabbed all HRTS validated shots with shot number >= 79000.

  * These are stored in dictionary format in a pickle file. If you have the file, then each key in the dictionary is a pulse number
  * Each pulse is another dicitonary with keys: `'inputs', 'outputs'`
  * Inputs is a dictionary, with keys corresponding to the control parameters
    * Each control parameters is a dictionary, with keys `'values', 'time'`
  * Outputs is a dictionary with keys `'NE', 'DNE', 'DTE', 'TE', 'radius', 'time'`
  * If you know you know


Data-splitting
""""""""""""""

For each pulse, we should take 70% of the profiles for training, 10% for validation, and 20% for testing. This will ensure that each pulse is represented in each dataset.

See above

Preprocessing and DataClasses
""""""""""""""""""""""""""""""""""""""""""

Currently, we just take the max density value for the training set and divide all ne points by that value. This constrains the input profiles to be between 0 and 1. This is subject to change.
The dataclasses are stored in :file:`src/data/profile_dataset.py`
