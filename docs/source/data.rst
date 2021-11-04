Data Storage
===================================

This subdirectory contains the various forms of datasets used in the analysis.


| data
| ├── raw_datasets < original immutable data dump from HEIMDALL
| |  ├── all_shots.pickle
| ├── processed  < final, canonical datasets for modeling



Raw Datasets
----------

We grab the inputs and outputs from JET PPFs and store them pickled python dictionary, `all_shots.pickle`.
Relevant JETDSP info:

- DDA: HRTS
- PPFUID: JETPPF
- Sequence \# 0 (**?**)
- Datatype: NE (density) DNE (error in density)
- T window range: (40-75s)
- **FLAGHRTS??**
- 63 X size, 701 T size

**Total HRTS validated shot count: 4942 Shots.**

- Maybe start looking at unvalidated?


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

- EFIT: shape (989,)
	- Q95
	- RGEO (Major R)
	- CRO (Minor radius = (Rmax-Rmin)/2 )
	- VOLM (plasma volume)
	- TRIU and TRIL (Upper and lower triangularity)
	- XPFL (limiter and divertor flag)
	- XIP (Plasma current measured)
	- ELON (Elongation)
	- POHM

The time resolution for each EFIT parameter is given as EFIT_T in the inputs dictionary.

Each of the next columns have both the value and time stored in it as a dicitionary with keys 'values', 'time'

- SCAL: shape (1024,)
	- BT
- GASH: shape (8101,)
	- ELER (electron flow rate)
- NBI: shape (13104,)
	- PTOT (Total Neutral Beam Power)
- ICRH: shape (1000,)
	- PTOT (Total ICRH power)



Output Profiles
~~~~~~~~~~~~

- Density (NE)
  - 2D array: shape (701, 63)
- Error (DNE)
  - 2D array: shape (701, 63)
- Temporal (time)
  - Temporal resolution of profile, shape (701,) corresponds to the rows of above profile array
  - This changes depending on the pulse
- Radial (radius)
  - Spatial resolution of profile
  - This changes depending on the pulse

Shape is (701, 63) for each pulse, where 701 and 63 are the temporal and spatial resolution respectively.



Processed
----------
We can convert the massive dictionary to ML readable formats.

There is a problem in that the temporal resolution of neped does not exactly align with that of the other input parameters.
So we need to do some windowing technique.
