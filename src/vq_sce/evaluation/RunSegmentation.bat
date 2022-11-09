@ECHO OFF

SET conda_path="C:\Users\roybo\Anaconda3\"
SET working_dir="C:\Users\roybo\Programming\PhD\007_CNN_Virtual_Contrast\"
SET expt_dir="syntheticcontrast_v02\evaluation\expts\"

CALL %conda_path%\Scripts\activate.bat base
cd %working_dir%

CALL %conda_path%\Scripts\activate.bat TF

SET times="0.0" "0.25" "0.5" "0.75" "1.0" "1.25" "1.5" "1.75" "2.0" "2.5" "3.0" "3.5" "4.0" "5.0" "10.0" "20.0"
SET models="2_save230" "2_save170_patch" "H2_save280" "H2_save300_patch"

SETLOCAL ENABLEDELAYEDEXPANSION
(FOR %%m IN (%models%) DO (
    SET model=%%m
    SET expt=%expt_dir:"=%!model:"=!
    (FOR %%t IN (%times%) DO (

        CALL python -m syntheticcontrast_v02.evaluation.segmentation_train -p !expt! -t %%t

    ))
))

PAUSE