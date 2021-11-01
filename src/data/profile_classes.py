from dataclasses import dataclass 
from typing import List 



@dataclass 
class Feature: 
    name: str
    time: List
    value: List 

@dataclass 
class Profile: 
    name: str
    time: List
    radius: List 
    value: List

@dataclass 
class Pulse: 
    inputs: List[Feature]
    outputs: List[Profile]

@dataclass 
class ProfileDataset: 
    profiles: List[Pulse]
