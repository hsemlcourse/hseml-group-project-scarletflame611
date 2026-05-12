.PHONY: lint lint-fix preprocess train test docker-up

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

preprocess:
	python src/preprocessing.py

train:
	python src/modeling.py

test:
	pytest tests/ -v

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

lint-init:
	pre-commit run --all-files