Dataset(s) in Profile Database
=================================

Within the profile database, there are various datasets that use the pulse information stored above. They are found in the top-level subgroup :code:`'processed_datasets'`.


This is subject to change, don't take my word for it! 
.. code-block:: text

  profile_database           - the database file, (if have it, you know)
  ├── 87500                  - Shot # as explained in data
  ├── 87501                  - Shot # as explained in data
  ...
  ├── processed_datasets     - final, canonical datasets for modeling
  │   ├── PSI22              - The current version of the PSI dataset




List Of Current
~~~~~~~~~~~~~~~

From the raw data, we create the following subsets of data. 
The name of the dataset is written :code:`'name'`

- :code:`'raw'` : Raw Signals + Mask + Radii

 - We have two options here, either make every single slice a dataset, that way it fits into the HDF5, or put things into a massive list dict... Not sure yet. 
 - Here we just take the raw time slices (cut between R = [-0.2, 0.05], i.e., pedestal regoin) for each pulse. Within the :code:`'raw'` group, there are subgroups, :code:`'all', 'train', 'valid', 'test'` (splitting procedure outlined elsewhere), which the naming convention is probably clear. The Te and Ne values are given as :code:`X`, and :code:`y` are the machine parameters per ususal. Additionally given within the subgroups, are the :code:`'mask', 'radii'`. The mask correpsonds to boolean arrays the length of the slice in :code:`'X'` that are True for all values except those in SOL (Rmid - Rmidsep > 0.0) where Te > 200 (see 2-point model page TBD). The radii are the corresponding raddi (Rmid - Rmidsep) for each time slice. 
 - The obvious benifit of this dataset is we are not introduing any uncertainties, as the values fed to any model are strictly coming from the measurements. 
 - **NB** Because we cut the signals to the pedestal region, the time slices vary in the amount of data contained in the pedestal regoin. For example, the time slices for each pulse may have 20-25 measurements in the pedestal region, but not necessarily constant amount of data. 

The structure in the HDF5 file is outlined below: 

.. code-block:: text

  processed_datasets         - the database file, (if have it, you know)
  ├── raw                    - Raw data
  │   ├── all                - All, i.e., no splitting into train, test, or valid 
  │   │   ├── X              - Density and temperature on a per-time-slice basis 
  │   │   ├── y              - Machine parameters
  │   │   ├── mask           - Boolean array for each time slice, denoting Te>200eV in SOL
  │   │   ├── radii          - The Rmid - Rmidsep values for each time slice
  │   ├── train              - Same as above, but for just training slices/pulses
  │   │   ├── ...            - Same X, y, mask, radii as above
  │   ├── valid              - Rinse repeat
  │   ├── test               - 
  ├── interp_raw             - See below
  ...



- :code:`'interp_raw'` : Interpolation of raw signals + Mask 

  - Here we interpolate the raw signals onto a common x-domain, hence why there is no Radii given for each slice, as it is common for all. The same subgroups apply from above, but an additional dataset will be given at the subgroup level, :code:`'radii'` which denotes the common x-domain for all the time slices. 
  - There are many reasons why we should probably not do this. And more research needs to be done on this. But hey, for the moment, we do it. 
