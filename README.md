# Moxie, a study on compressing the plasma state

We study VAEs and how they can be used to compress information from diagnostic data into a representation of the plasma state.


## Building data (necessary)

In order to gather data for usage, you need a JET account. If you have access to JET and Heimdall/NoMachine, then you can continue.

#### The short way

Since you have access to NoMachine, then you can simply collect the data from my personal folder using the following commands:

1. In your VNC/Virtual Machine within NoMachine, clone this repository.
2. run `cd moxie/data/ && chmod u+x make_all.bash && ./collect_all.bash`
3. Some files should appear the main `moxie/data` folder.

#### The long way
If you want to rebuild the datasets from scratch, then
The library requirements for the data pulling are: pandas, numpy, tqdm and SAL from JET. I suggest that you make a virtual environment and install this packages separately.

You also need access to the JET pedestal database. This can be found on the T17-05 wiki.

1. In your VNC/Virtual Machine within NoMachine, clone this repository.
2. run `cd moxie/data/ && chmod u+x make_all.bash && ./make_all.bash`
3. Some files should appear the main `moxie/data` folder.


## Usage & Installation

In order to do any deep learning shenanigans, you need to 1.) have the data (see above) and 2.) install this package. You probably shouldn't run any models on NoMachine, so do this on your local machine or on a cluster (instructions TBD).

Assuming you have some virtual environement, activate it and follow the steps:

1. clone this repository and `cd` into moxie.
2. Install the main `moxie` package by running `pip install -e .` within the repo dir (i.e., directory with `setup.py`)


Once installing, you can make your own experiment in `experiments/`.  
See `experiments/PSI_22/` for an example experiment.
- `run.py`
- `search.py`
