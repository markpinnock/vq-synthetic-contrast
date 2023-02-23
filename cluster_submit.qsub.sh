#!/bin/bash
#$ -S /bin/bash
#$ -l gpu=true
#$ -l h_rt=96:00:00
#$ -l tmem=16G
#$ -N expt_name
#$ -wd /path/to/save/dir

source /share/apps/source_files/python/python-3.10.0.source
source /share/apps/source_files/cuda/cuda-11.2.source

FOLDER=/path/to/wd
EXPT=$1

cd $FOLDER
source contrast/bin/activate

train -p expts/$EXPT