import os
from pathlib import Path

expt_path = Path("/path/to/experiments")
expts = expt_path.glob("*")

for expt in expts:
    for pred in expt.glob("predictions-*"):
        os.system(
            f'calc_metrics -p "{str(pred)}" -d /path/to/image/data -su validation',
        )
