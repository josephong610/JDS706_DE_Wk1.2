install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
format:
	black data.py data_test.py

lint:
	flake8 --ignore=C,N data.py data_test.py

test:
	python3 -m pytest -vv --cov=data data_test.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test