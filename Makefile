IMAGE_NAME = ids706_project

# ----------------------------
# Local development workflow
# ----------------------------
install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black data.py data_test.py

lint:
	flake8 --ignore=E501,F401 data.py data_test.py

test:
	python3 -m pytest -vv data_test.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test clean


# ----------------------------
# Dockerized workflow
# ----------------------------
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-test: docker-build
	docker run --rm $(IMAGE_NAME) pytest -vv --disable-warnings -q data_test.py

docker-format: docker-build
	docker run --rm $(IMAGE_NAME) black --check data.py data_test.py

docker-lint: docker-build
	docker run --rm $(IMAGE_NAME) flake8 --ignore=E501,F401 data.py data_test.py

docker-clean: docker-build
	docker run --rm $(IMAGE_NAME) rm -rf __pycache__ .pytest_cache .coverage

docker-shell: docker-build
	docker run -it --rm $(IMAGE_NAME) bash

docker-all: docker-build docker-format docker-lint docker-test docker-clean
