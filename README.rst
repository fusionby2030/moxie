# moxie

## Pedestal Profile 
The profile could come in many shapes and sizes, but an individual profile is defined here as: 
a time averaged slice of the pedestal density (or temperature) within the domain Psi-n = {0.75, 1.05}. 

A shot consists of many profiles, i.e., a time evolving profile.  

This profile is captured using HRTS diagnostics. Normally, it is fit using a MTANH function (see Lorenzo, et al. 2020). 

### Strohman Analysis

We could predict the Raw HRTS data or MTANH fit of a profile using: 
- a) scalar inputs
- b) time series 


#### Datagathering


We need to get the inputs/outputs from JET DB. The shots in question are from pulse number >79000 (where HRTS system was upgraded).

---

##### Profile  (Output)
The output is of course the pedestal profile, but the trick will be getting the various time averages. 
 
- HRTS Resolution [Source](https://users.euro-fusion.org/ekwiki/index.php/KE11_High_Resolution_Thomson_Scattering_(HRTS))
	- **Spatial**: Currently HRTS provides 63 spatial data points per profile, with a 20 Hz repetition rate for the duration of a JET pulse. The system has a spatial resolution of ~ 1.6 cm in the core region and ~ 1 cm in the pedestal region. A total mapping is for Rmid ~  3.0 to 3.9. 
	- **Time**: laser operates at 20 Hz (Q-switched Nd:YAG, 3J per 20 ns pulse at 1064 nm) and fires 800 laser pulse, starting just before the plasma is initiated (to give us stray-light data). This means that the HRTS data typically spans the entire JET pulse from t=40 - 75 s at a sampling frequency of 20 Hz.
	- **Additional Info**: HRTS measures scattered light by plasma. Amount of scattered light is propto electron density. Width of scattered spectra is propto the electron momentum (and thus Te). 

- What is difference between unfiltered density (datatype: NNE) vs density (datatype: NE)
	- [WIKI Description](https://users.euro-fusion.org/ekwiki/index.php/HRTS_data_guide): In SOL and late pedestal regions there are spikes in the HRTS scans, since there is very little data there in the first place. The raw values are stored in NNE, but when it spikes in SOL, the value in NE is set to 1e-20, with error 1e20. 
	- We will take the filtered value 

- If we ever want to get more points in SOL and Pedestal, we could use the lithium beams, but this is not available for all shots.
	- **IDEA**: Use the data from lithium beam to train model, then for those without, try to generate the lithium beam profile! 
	

###### Relevant JETDSP info
For gathering the density: 
- DDA: HRTS 
- PPFUID: JETPPF 
- Sequence \# 0 (**?**) 
- Datatype: NE (density) DNE (error in density) 
- T window range: (40-75s)
- **FLAGHRTS??**
- 63 X size, 701 T size

--- 

##### Main Eng (Inputs)

List of DDA's DataType associated with them. 
- EFIT: 989 points per shot. (Could also be called EFTP) 
	- Q95
	- RGEO (Major R)
	- CRO (Minor radius = (Rmax-Rmin)/2 )
	- VOLM (plasma volume) 
	- TRIU and TRIL (Upper and lower triangularity) 
	- XPFL (limiter and divertor flag)
	- XIP (Plasma current measured) 
	- ELON (Elongation) 
	- POHM

- SCAL: 1024 Points
	- BT  

- GASH: 8101 points per shot
	- ELER (electron flow rate)
	- CGAS (Gas codes**?**)
- NBI: 13104 points per shot
	- PTOT (Total Neutral Beam Power)
- ICRH: (1000 Points per shot)
	- PTOT (Total ICRH power)
 

---

Gathering Process: 
1.) Check HRTS Flag of shot: `flag, irrelevant = ppfgsf(shot,0,'HRTS','NE',mxstat=200)`
	- if flag == 0, skip shot
2.) Get HRTS Data
3.) Get Input Data

```
data,x,t,nd,nx,nt,dunits,xunits,tunits,desc,comm,seq,ier = ppfdata(shot,dda,dtyp,seq=0,uid="jetppf",device="JET", fix0=0,reshape=0,no_x=0,no_t=0,no_data=0)
```

The important shit is stored in data, t, 




#### Datastorage 

- Python Classes
	- Then dump to JSON! 
- See data/ dir
