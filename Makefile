# ==============================================================================
# Basic RAG FastAPI Project Makefile
# ==============================================================================

.DEFAULT_GOAL := help
.ONESHELL:
SHELL := /bin/bash

# ------------------------------------------------------------------------------
# Configurable Environment Variables (override via `make <target> VAR=value`)
# ------------------------------------------------------------------------------
APP            ?= app.main:app
HOST           ?= 127.0.0.1
PORT           ?= 8000
UVICORN_FLAGS  ?= --reload

# Tool Wrappers using uv
UV             := uv run
PYTHON         := $(UV) python
UVICORN        := $(UV) uvicorn
ALEMBIC        := $(UV) alembic
RUFF           := $(UV) ruff
PYTEST         := $(UV) pytest
BASEDPYRIGHT   := $(UV) basedpyright
PRECOMMIT      := $(UV) pre-commit

RUFF_CHECK     := $(RUFF) check
RUFF_FORMAT    := $(RUFF) format
DOCKER_COMPOSE := docker compose

.PHONY: help \
        env sync install \
        dev run shell \
        format format-check lint lint-fix typecheck check \
        test test-unit test-cov verify ci precommit \
        migrate makemigrations downgrade current history \
        docker-up docker-down docker-logs \
        clean clean-pyc clean-cache

# ==============================================================================
# HELP & UTILITIES
# ==============================================================================

## help: Display categorized list of available targets with descriptions
help:
	@echo "Basic RAG Makefile Commands"
	@echo "==========================="
	@echo "Usage: make [target] [VAR=value]"
	@echo ""
	@echo "Environment & Setup:"
	@echo "  env                Create local .env file from .env.example"
	@echo "  sync               Synchronize dependencies with uv sync"
	@echo "  install            Alias for sync"
	@echo ""
	@echo "Development:"
	@echo "  dev                Start FastAPI dev server (configurable: HOST, PORT, APP)"
	@echo "  run                Alias for dev"
	@echo "  shell              Start interactive Python REPL"
	@echo ""
	@echo "Quality & Formatting:"
	@echo "  format             Format all Python files using Ruff"
	@echo "  format-check       Check formatting without applying changes"
	@echo "  lint               Run Ruff code linter"
	@echo "  lint-fix           Fix auto-fixable lint issues"
	@echo "  typecheck          Run static type check using BasedPyright"
	@echo "  check              Run format-check, lint, and typecheck"
	@echo ""
	@echo "Testing & Verification:"
	@echo "  test               Run Pytest suite"
	@echo "  test-unit          Run unit tests in tests/ directory"
	@echo "  test-cov           Run Pytest suite with coverage report"
	@echo "  verify             Run full verification suite (check + test)"
	@echo "  ci                 Run CI pipeline target (check + test)"
	@echo "  precommit          Run all pre-commit hooks on repository files"
	@echo ""
	@echo "Database & Migrations:"
	@echo "  migrate            Apply pending migrations (alembic upgrade head)"
	@echo "  makemigrations     Generate migration revision (requires m=\"description\")"
	@echo "  downgrade          Rollback last migration step"
	@echo "  current            Display current migration revision"
	@echo "  history            Display migration history"
	@echo ""
	@echo "Docker Infrastructure:"
	@echo "  docker-up          Start local database infrastructure"
	@echo "  docker-down        Stop local database infrastructure"
	@echo "  docker-logs        Stream database logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean              Clean all Python bytecode and test cache artifacts"
	@echo "  clean-pyc          Clean compiled Python files (__pycache__, .pyc)"
	@echo "  clean-cache        Clean tool cache directories (.pytest_cache, .ruff_cache)"

## env: Create local .env file from .env.example if it does not exist
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from .env.example"; \
	else \
		echo ".env file already exists"; \
	fi

## sync: Synchronize project environment and dependencies with uv
sync:
	uv sync

## install: Alias for sync target
install: sync

# ==============================================================================
# DEVELOPMENT
# ==============================================================================

## dev: Start FastAPI development server with auto-reload
dev:
	$(UVICORN) $(APP) $(UVICORN_FLAGS) --host $(HOST) --port $(PORT)

## run: Alias for dev target
run: dev

## shell: Start interactive Python REPL within project environment
shell:
	$(PYTHON)

# ==============================================================================
# CODE QUALITY & LINTING
# ==============================================================================

## format: Format all Python files in the repository using Ruff
format:
	$(RUFF_FORMAT) .

## format-check: Verify that all Python files adhere to formatting rules
format-check:
	$(RUFF_FORMAT) --check .

## lint: Check codebase for code quality and style issues with Ruff
lint:
	$(RUFF_CHECK) .

## lint-fix: Automatically fix auto-fixable linting issues with Ruff
lint-fix:
	$(RUFF_CHECK) --fix .

## typecheck: Run static type checks using BasedPyright
typecheck:
	$(BASEDPYRIGHT)

## check: Run formatting checks, linter verification, and type checking
check: format-check lint typecheck

# ==============================================================================
# TESTING & VERIFICATION
# ==============================================================================

## test: Run complete Pytest test suite
test:
	$(PYTEST)

## test-unit: Run unit tests under the tests/ directory
test-unit:
	$(PYTEST) tests/

## test-cov: Run Pytest suite with terminal coverage report
test-cov:
	$(PYTEST) --cov=app --cov-report=term-missing

## verify: Run complete local pre-commit check (check + test)
verify: check test

## ci: Standard entry point for CI workflows (check + test)
ci: check test

## precommit: Run all pre-commit hooks across the repository
precommit:
	$(PRECOMMIT) run --all-files

# ==============================================================================
# DATABASE & MIGRATIONS (ALEMBIC)
# ==============================================================================

## migrate: Apply all pending database migrations
migrate:
	$(ALEMBIC) upgrade head

## makemigrations: Generate a new Alembic migration revision (usage: make makemigrations m="description")
makemigrations:
	@if [ -z "$(m)" ]; then \
		echo "Error: Migration message required. Usage: make makemigrations m=\"your description\""; \
		exit 1; \
	fi
	$(ALEMBIC) revision --autogenerate -m "$(m)"

## downgrade: Rollback the single most recent migration step
downgrade:
	$(ALEMBIC) downgrade -1

## current: Display the current active database migration revision
current:
	$(ALEMBIC) current

## history: Display complete database migration history
history:
	$(ALEMBIC) history

# ==============================================================================
# DOCKER INFRASTRUCTURE
# ==============================================================================

## docker-up: Start local database infrastructure (Compose or standalone container)
docker-up:
	@if [ -f docker-compose.yml ] || [ -f compose.yaml ]; then \
		$(DOCKER_COMPOSE) up -d; \
	elif docker ps -a --format '{{.Names}}' | grep -q "^pg-local$$"; then \
		docker start pg-local; \
	else \
		docker run -d --name pg-local -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=basic_rag pgvector/pgvector:pg15; \
	fi

## docker-down: Stop local database infrastructure
docker-down:
	@if [ -f docker-compose.yml ] || [ -f compose.yaml ]; then \
		$(DOCKER_COMPOSE) down; \
	else \
		docker stop pg-local 2>/dev/null || true; \
	fi

## docker-logs: Stream logs from local database infrastructure
docker-logs:
	@if [ -f docker-compose.yml ] || [ -f compose.yaml ]; then \
		$(DOCKER_COMPOSE) logs -f; \
	else \
		docker logs -f pg-local; \
	fi

# ==============================================================================
# CLEANING
# ==============================================================================

## clean-pyc: Remove Python compiled bytecode files and __pycache__ directories
clean-pyc:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

## clean-cache: Remove test and tool cache directories (.pytest_cache, .ruff_cache)
clean-cache:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

## clean: Remove bytecode files and cache artifacts
clean: clean-pyc clean-cache
