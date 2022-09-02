@ECHO OFF

SET conda_path="C:\Users\roybo\Anaconda3\"
SET working_dir1="C:\Users\roybo\Programming\PhD\007_CNN_Virtual_Contrast\"
SET working_dir2="C:/Users/roybo/Programming/PhD/007_CNN_Virtual_Contrast"
SET save_dir="C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output"
SET data_dir="D:/ProjectImages/SyntheticContrastNeedle/"

CALL %conda_path%\Scripts\activate.bat base
cd %working_dir1%

SET model="2_save230"
SET expt="test_pix2pix/%model%"
CALL %conda_path%\Scripts\activate.bat TF
CALL python -m syntheticcontrast_v02.inference -p %expt% -d %data_dir% -f "both" -s
CALL %conda_path%\Scripts\activate.bat nrrd
CALL python -m syntheticcontrast_v02.preproc.postprocess -p %working_dir2%/%expt%/predictions -s %save_dir%/%model%/Needle

SET model="H2_save280"
SET expt="test_pix2pix/%model%"
CALL %conda_path%\Scripts\activate.bat TF
CALL python -m syntheticcontrast_v02.inference -p %expt% -d %data_dir% -f "both" -s
CALL %conda_path%\Scripts\activate.bat nrrd
CALL python -m syntheticcontrast_v02.preproc.postprocess -p %working_dir2%/%expt%/predictions -s %save_dir%/%model%/Needle

SET model="H2_save300_patch"
SET expt="test_pix2pix/%model%"
CALL %conda_path%\Scripts\activate.bat TF
CALL python -m syntheticcontrast_v02.inference -p %expt% -d %data_dir% -f "both" -s
CALL %conda_path%\Scripts\activate.bat nrrd
CALL python -m syntheticcontrast_v02.preproc.postprocess -p %working_dir2%/%expt%/predictions -s %save_dir%/%model%/Needle

PAUSE