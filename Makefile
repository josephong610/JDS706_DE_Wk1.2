IMAGE_NAME = ids706_project

# Local development
install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black data.py data_test.py

lint:
	flake8 --ignore=C,N data.py data_test.py

test:
	python3 -m pytest -vv --cov=data data_test.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

# Dockerized workflow
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-test: docker-build
	docker run --rm $(IMAGE_NAME)

docker-shell: docker-build
	docker run -it --rm $(IMAGE_NAME) bash

# Run everything locally
all: install format lint test

# Run everything inside Docker
docker-all: docker-build docker-test
