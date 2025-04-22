install:
	uv pip sync pyproject.toml --features dev

test:
	# python -m pytest --nbval *.ipynb
	python -m pytest -vv --cov= .


format:	
	black .

lint:
	ruff check --fix .

deploy:
	# no rules for now
		
all: install lint test format
