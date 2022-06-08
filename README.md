# Moxie, a study on compressing the plasma state

We study VAEs and how they can be used to compress information from diagnostic data into a representation of the plasma state.

To view an overview of what is capable, see `POSTER.pdf`, a poster from the 25th International Conference on Plasma Surface Interactions in Controlled Fusion Devices (PSI-25).

# Citation

TBD

## Building data (necessary)

In order to gather data for usage, you need a JET account. If you have access to JET and Heimdall/NoMachine, then you can continue.

#### The short way (TRY THIS FIRST)

Since you have access to NoMachine, then you can simply collect the data from my personal folder using the following commands:

1. In your VNC/Virtual Machine within NoMachine, clone this repository.
2. run `cd moxie/data/ && chmod u+x make_all.bash && ./collect_all.bash`
3. Some files should appear in the `moxie/data/` folder.

#### The long way
If you want to rebuild the datasets from scratch, then
The library requirements for the data pulling are: pandas, numpy, tqdm and SAL from JET. I suggest that you make a virtual environment and install this packages separately.

You also need access to the JET pedestal database. This can be found on the T17-05 wiki.

1. In your VNC/Virtual Machine within NoMachine, clone this repository and `cd` into the main directory.
2. run `cd moxie/data/ && chmod u+x make_all.bash && ./make_all.bash`
3. Some files should appear the main `moxie/data` folder.

### If success

If you achieved either of the two options above, you will now have 4 files within the directory `moxie/data`, namely:
1. `proccessed/ML_READY_dict_07062022.pickle`
2. `proccessed/processed_pulse_dict_07062022.pickle`
3. `raw/jet-pedestal-database.csv`
4. `raw/JET_RAW_DATA_07062022.pickle`

Now you will now need to transfer the files from `processed` to your working computer (should probably not train ANNs on NoMachine) however you normally transfer files from NoMachine. You can do either of the following:

1. Clone the repostiory again on your working computer, `cd` into the directory, and move the NoMachine directory you made above, `moxie/data`, manually, perserving file structure, or
2. Simply copy the entire `moxie` folder you have cloned previously into NoMachine.

This will take some time, as it is a large amount of data being transfered. Grab a tea.

## Usage & Installation

In order to do any deep learning shenanigans, you need to 1.) have the data (see above) and 2.) install this package.

Assuming you have some virtual environement and have already cloned this repository:

0. `cd` into the cloned directory.  
1. Install dependencies via: `python -m pip install -r --user requirements.txt`
2. Install the main `moxie` package by running `pip install -e . --user` within the main directory (i.e., directory with `setup.py`, this should be what you just `cd` into)
3. To test everything works, run `cd experiments/PSI_22 && python3 run.py`

Once installing, you can make your own experiment in `experiments/`.  
See `experiments/PSI_22/` for an example experiment and further details.
