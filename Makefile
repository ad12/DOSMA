autoformat:
	dev/linter.sh format

lint:
	dev/linter.sh

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

all: test build-docs