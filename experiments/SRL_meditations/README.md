# Meditations on SRL 

## Goal: Get a forward model of AUG data going 


### Data: 
    
- Combine dataclasses with torch dataset -> 
- Very crude method shown in `torch_dataset.py` and `python_dataset.py`
- Can go from raw AUG data -> ML readable. 
- train-val-test split of pulses 
- It passes both mps and profiles 
- TODO: Docs :P

### Model 

- Currently it is the E2C Model without any machine parameters passing through and no action. 
- TODO: implement machine parameter passing/decoding! 
    - Conditional priors and aux reg. _a la_ DIVA. 
- TODO: Action 
    - Should actually be the delta between t0 and t1 machine parameters. 