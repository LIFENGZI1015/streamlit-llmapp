# Makefile for python code
# 
# > make help
#
# The following commands can be used.
#
# init:  sets up environment and installs requirements
# install:  Installs development requirments
# format:  Formats the code with autopep8
# lint:  Runs flake8 on src, exit if critical rules are broken
# clean:  Remove build and cache files
# env:  Source venv and environment files for testing
# leave:  Cleanup and deactivate venv
# test:  Run pytest
# run:  Executes the logic
# https://gist.githubusercontent.com/MarkWarneke/2e26d7caef237042e9374ebf564517ad/raw/08348012fd4fc23681731145a14d52e21afbf1a8/Makefile

VENV_PATH='env/bin/activate'
ENVIRONMENT_VARIABLE_FILE='.env'

env: ## Source venv and environment files for testing
env:
	python3 -m venv env
	source $(VENV_PATH)
	source $(ENVIRONMENT_VARIABLE_FILE)

install: ## Installs development requirements
install:
	pip install --upgrade pip
	pip install black pylint pytest

init: ## Installs requirements
init:
	pip install -r requirements.txt

format: ## Formats the code with autopep8 
format:
	find . -name '*.py' -exec black {} +

lint: ## Runs flake8 on src, exit if critical rules are broken
lint:
	pylint --disable=R,C,W0702,W0621,W0611 ./*.py
# test: ## Run pytest
# test:
# 	python -m pytest -vv --cov=test

run: ## Run app using streamlit
run:
	streamlit run app_llmchat.py
# quit:
# quit:
# 	^C

clean: ## Remove build and cache files
clean:
	rm -rf .pytest_cache