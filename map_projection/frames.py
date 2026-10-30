"""Immutable render requests; importing these does not load a graphics driver."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SunScene:
    key: str
    z: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    terrain_size: tuple
    events: object = None


@dataclass(frozen=True)
class SunFrame:
    scene: SunScene
    sun: np.ndarray
    ambient: float
    bias: float
    time: str
    enabled: bool = True
    method: str = "fast"
    layers: object = None
    light_gain: float = 1.
    light_floor: float = 0.
    light_gamma: float = 1.

    @property
    def shape(self):return (*self.scene.uv.shape[:2],3)
