install:
	uv pip sync uv.lock

lock:
	uv pip compile pyproject.toml --all-features -o uv.lock

test:
	python -m pytest -vv --cov= .

format:	
	black .

lint:
	ruff check --fix .

deploy:
	# no rules for now

all: lock install lint test format
