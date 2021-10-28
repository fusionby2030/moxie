# Data Storage

After downloading the data, we store it in the following format: 

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


