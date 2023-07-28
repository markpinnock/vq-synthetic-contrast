@ECHO OFF

SET conda_path="C:\path\to\miniconda3\"
SET working_dir="C:\path\to\experiments\"
SET data_dir="D:\\path\\to\\data\\"

CALL %conda_path%\Scripts\activate.bat base
CALL %conda_path%\Scripts\activate.bat contrast
cd %working_dir%

@REM for %%x in (
@REM     "ce_ss_vq128"
@REM     ) do (
@REM         CALL calc_metrics -p %%x -d %data_dir% -s contrast
@REM         )

@REM for %%x in (
@REM     "sr_ss_vq512"
@REM     ) do (
@REM         CALL calc_metrics -p %%x -d %data_dir% -s super_res
@REM         )

@REM for %%x in (
@REM     "jo_ss_vq128"
@REM     ) do (
@REM         CALL calc_metrics -p %%x -d %data_dir% -s super_res
@REM         CALL calc_metrics -p %%x -d %data_dir% -s contrast
@REM         )

CALL calc_metrics -p "ce_vq8" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq16" -d %data_dir% -st "contrast" -ep "1700"
CALL calc_metrics -p "ce_vq16" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq32" -d %data_dir% -st "contrast" -ep "1600"
CALL calc_metrics -p "ce_vq32" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq64" -d %data_dir% -st "contrast" -ep "1600"
CALL calc_metrics -p "ce_vq64" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq128" -d %data_dir% -st "contrast" -ep "1600"
CALL calc_metrics -p "ce_vq128" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq256" -d %data_dir% -st "contrast" -ep "1600"
CALL calc_metrics -p "ce_vq256" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq512" -d %data_dir% -st "contrast" -ep "2000"
CALL calc_metrics -p "ce_vq1024" -d %data_dir% -st "contrast" -ep "1600"
CALL calc_metrics -p "ce_vq1024" -d %data_dir% -st "contrast" -ep "2000"
