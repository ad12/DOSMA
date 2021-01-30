autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8

test:
	set -e
	coverage run -m pytest tests/

build-docs:
	set -e
	rm -rf docs/build/generated
	cd docs && make html

dev:
	pip install black coverage isort flake8
	pip install sphinx sphinx-rtd-theme recommonmark
	pip install -r docs/requirements.txt

all: autoformat test build-docs