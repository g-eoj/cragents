.PHONY: format
format: ## Format the code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint the code
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck-pyright
typecheck-pyright:
	@# PYRIGHT_PYTHON_IGNORE_WARNINGS avoids the overhead of making a request to github on every invocation
	PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright

.PHONY: typecheck
typecheck: typecheck-pyright ## Run static type checking

.PHONY: test
test: ## Run tests and collect coverage data
	COLUMNS=150 uv run coverage run -m pytest -n auto --dist=loadgroup --durations=20
	uv run coverage combine
	uv run coverage report
