@ECHO OFF

SET conda_path="C:\path\to\miniconda3\"
SET working_dir="C:\path\to\experiments\"
SET data_dir="D:\\path\\to\\data\\"

CALL %conda_path%\Scripts\activate.bat base
CALL %conda_path%\Scripts\activate.bat contrast
cd %working_dir%

for %%x in (
    "ce_ss_vq128"
    ) do (
        CALL calc_metrics -p %%x -d %data_dir% -s contrast
        )

for %%x in (
    "sr_ss_vq512"
    ) do (
        CALL calc_metrics -p %%x -d %data_dir% -s super_res
        )

for %%x in (
    "jo_ss_vq128"
    ) do (
        CALL calc_metrics -p %%x -d %data_dir% -s super_res
        CALL calc_metrics -p %%x -d %data_dir% -s contrast
        )
