install:

install-dev: install
	pip3 install ruff==0.0.286

lint:
	ruff *.py --select W,C9,E,F --ignore E501

test:
	python3 -m unittest discover
