env:
	conda env create -f environment.yml

cluster_env:
	conda env create -f cluster_environment.yml

pip_local:
	pip install -e .

train:
	export CUDA_VISIBLE_DEVICES=$(GPU) && train --expt_path $(EXPT_PATH)

predict:
	predict --expt_path $(EXPT_PATH)