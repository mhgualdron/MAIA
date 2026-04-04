.PHONY: lintfix lintcheck unittest

# ======================
# Formatea y arregla el código
# ======================
lintfix:
	poetry install
	poetry run black .
	poetry run isort . --profile black
	poetry run bandit -c pyproject.toml -r .
	poetry run ruff check --fix

# ======================
# Solo revisa estilo y seguridad, no modifica nada
# ======================
lintcheck:
	poetry install
	poetry run black --check .
	poetry run isort --check . --profile black
	poetry run bandit -c pyproject.toml -r .
	poetry run ruff check

# ======================
# Corre tests con cobertura
# ======================
unittest:
	poetry install
	poetry run pytest --cov=src -v -s --cov-fail-under=70 --cov-report term-missing