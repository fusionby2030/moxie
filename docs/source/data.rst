# Data Storage

Unfortunately, the data is at the moment runing out of memory. Idea is to store it in a database file? 

This subdirectory contains the various forms of datasets used in the analysis.

```
|---data
|    |---raw_datasets < original immutable data dump from HEIMDALL
|    |---processed  < final, canonical datasets for modeling
```
---


## Raw Datasets

We grab the inputs and outputs from JET PPFs and store them pickled python dictionary, that is compressed with gzip.

To take out the raw file, use gzip to uncompress, then pickle load the dictionary which has the following format:

**Total HRTS validated shot count: 4942 Shots.**
- Maybe start looking at unvalidated?



```
{'pulse_1': {'inputs': {'BT': {'values': np.array, 'time': np.array},
                        'IP': {'values': np.array, 'time': np.array}, ...}
            'outputs': {'time': {'R': np.array, 'neped'}}
            },
'pulse_2': {'inputs': {'BT': {'values': np.array, 'time': np.array},
                        'IP': {'values': np.array, 'time': np.array}, ...}
            'outputs': {'time': {'R': np.array, 'neped'}}
            },
            }
```

---
**Input Columns**
`'Q95', 'RGEO', 'RCO', 'VOLM', 'TRIU', 'TRIL', 'XIP', 'ELON', 'BTAX', 'POHM', 'P_NBI', 'P_ICRH', 'ELER'`

---
**Output Profiles**

The Error (DNE) and Value (NE) are stored in a 3D array, of dim time, spataial, and value.

---


## Processed

We can convert the HDF5 file into these abstract classes for easy ML feeding.


##### The Dataset class
- Shall hold all of the pulses

```
class Dataset:
	self.pulses: [List of Pulse Objects]

	TBD!
	flag: 'train', or 'val', or 'test'

	def __len__(self):
		return self.pulses
	def __getitem__(self, idx):
		return self.pulses[idx].inputs, pulses[idx].outputs

```

##### The Pulse class
- Shall hold the inputs and outputs

```
class Pulse:
	inputs: (List[Input Objects])
	outputs: (List[Output Objects])
	number: (int) the shot number associated with the pulse
```

##### The Input Class
- Shall hold the data for the given input for each time
- Shall have a function that can average over time steps
```
class Input:
	name: (str) the id of the input, e.g., 'Ip(MA)'
	values_t: (List[Tuple(float, float)]) where (float, float) > (value, time)
	values: List[float]
	t: List[float]

```
- The values could be split into another class


##### The Output Class

- Shall be able to return either a single profile for given time step
- Shall be able to return the profiles for all time steps
```
class Output:
	name: (str) the id of the output, e.g., 'density'
	profiles: (List[Profile Objects]) the corresponding profile classes
	times: (List[floats]) corresponding times for the profiles
```

##### The Profile Class

```
class Profile:
	r_values: (List[float]) the radius
	density: (List[float]) the density for each given radius
	time_step: (float) the timestep associated with the profile
```
