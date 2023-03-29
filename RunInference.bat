@ECHO OFF

SET conda_path="C:\path\to\miniconda3\"
SET working_dir="C:\path\to\experiments\"
SET data_dir="D:\\path\\\to\\data\\"
SET original_dir="Z:\\path\\to\\original\\data\\"

CALL %conda_path%\Scripts\activate.bat base
CALL %conda_path%\Scripts\activate.bat contrast
cd %working_dir%

for %%x in (
    "ce_ss_vq128"
    ) do (
        CALL predict -p %%x -d %data_dir% -o %original_dir% -s contrast -op save
        )

for %%x in (
    "sr_ss_vq512"
    ) do (
        CALL predict -p %%x -d %data_dir% -o %original_dir% -s super_res -op save
        )

for %%x in (
    "jo_ss_vq128"
    ) do (
        CALL predict -p %%x -d %data_dir% -o %original_dir% -s contrast -op save
        CALL predict -p %%x -d %data_dir% -o %original_dir% -s super_res -op save
        )
