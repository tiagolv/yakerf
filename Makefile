install:
	uv pip sync --features dev

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
