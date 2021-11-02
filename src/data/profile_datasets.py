from dataclasses import dataclass
from typing import List

"""
Scalar Dataset vs Time Series Dataset
Start with scalar average?
"""


@dataclass
class PulseDataset:
    name: str
    pulses: List[Pulse]

@dataclass
class Pulse:
    name: str # Pulse Number
    profiles: List[Profile]
    delta_T: float # This is how big the windows are for the features
    features: List[Feature]
    radius: List # The radial resolution of the HRTS scan

@dataclass
class Feature:
    values: List

@dataclass
class Profile:
    values: List
    time: float # The time value for the given profile
