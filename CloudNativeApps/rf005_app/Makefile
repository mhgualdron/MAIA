.PHONY: lintfix lintcheck unittest

lintfix:
	poetry install
	poetry run black .
	poetry run isort . --profile black
	poetry run bandit -c pyproject.toml -r .
	poetry run ruff check --fix

lintcheck:
	poetry install
	poetry run black --check .
	poetry run isort --check . --profile black
	poetry run bandit -c pyproject.toml -r .
	poetry run ruff check

unittest:
	poetry install
	poetry run pytest --cov=app -v -s --cov-fail-under=70 --cov-report term-missing
