.PHONY: lint lint-fix preprocess train test api static docker-up docker-down lint-init

lint:
	ruff check src/ tests/ api/

lint-fix:
	ruff check src/ tests/ api/ --fix
	ruff format src/ tests/ api/

preprocess:
	python src/preprocessing.py

train:
	python src/modeling.py

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --reload --port 8000

static:
	python -m http.server 8080 --directory static

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

lint-init:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install