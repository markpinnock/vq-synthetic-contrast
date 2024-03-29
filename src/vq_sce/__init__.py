from typing import TypedDict


class AbdoWindowKwargs(TypedDict):
    vmin: int
    vmax: int


ABDO_WINDOW: AbdoWindowKwargs = {"vmin": -150, "vmax": 250}
HQ_SLICE_THICK = 1
HU_MIN = -500
HU_MAX = 2500
IMAGE_HEIGHT = 512
LQ_DEPTH = 3
LQ_SLICE_THICK = 4
MIN_HQ_DEPTH = 12
RANDOM_SEED = 5
