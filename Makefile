.PHONY: conda_env, cluster_env, pip_env, install_pip_req, install_pip_local, help

#-------------------------------------------------------------------------
# Conda Commands
#-------------------------------------------------------------------------

conda_env: ## Create conda environment
	conda env create -f environment.yml --force


#-------------------------------------------------------------------------
# Pip Commands
#-------------------------------------------------------------------------

pip_env:
	(source /share/apps/source_files/python/python-3.10.0.source)
	python3 -m venv contrast-env

install_pip_req:
	contrast-env/bin/python3 -m pip install -r requirements.txt

install_pip_local:
	contrast-env/bin/python3 -m pip install -e .

cluster_env: pip_env install_pip_req install_pip_local ## Create pip virtual environment and install package locally


#-------------------------------------------------------------------------
# Self-documenting Commands
#-------------------------------------------------------------------------

.DEFAULT_GOAL := help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
