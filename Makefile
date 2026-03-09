PYEXEC ?= "python3"


lint:
	$(PYEXEC) -m flake8 --ignore E251,E501 src

format:
	yapf -i -r -vv .
	isort --ls --ds .

clear-dspy-cache:
	rm -rf ~/.dspy_cache/

clean:
	rm -I slurm-*.out