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
	mkdir -p docs/source/_static
	rm -rf docs/build
	rm -rf docs/source/generated
	cd docs && make html

dev:
	pip install black coverage isort flake8
	pip install sphinx sphinx-rtd-theme recommonmark m2r2
	pip install -r docs/requirements.txt

all: autoformat test build-docs