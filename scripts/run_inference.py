import os
from pathlib import Path

expt_path = Path("/path/to/experiments")
expts = expt_path.glob("*")

for expt in expts:
    os.system(
        f'predict -p "{str(expt)}" -d /path/to/image/data -o /path/to/original/data'
        f"-st contrast -su validation -ep 2000 -m 4 -op save",
    )
