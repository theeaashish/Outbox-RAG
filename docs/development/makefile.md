# Development Makefile Reference

This document serves as the developer guide for the repository's `Makefile`. The `Makefile` provides a unified, cross-platform CLI interface for running development servers, linters, static typecheckers, test suites, database migrations, and container operations.

---

## Tooling Overview

All Python commands inside the `Makefile` are wrapped with [`uv run`](https://docs.astral.sh/uv/) to ensure execution within the isolated project virtual environment without requiring manual activation.

---

## Configurable Variables

The `Makefile` supports dynamic variable overrides using the `?=` syntax:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP` | `app.main:app` | FastAPI application instance import path |
| `HOST` | `127.0.0.1` | Development server bind IP address |
| `PORT` | `8000` | Development server HTTP port |
| `UVICORN_FLAGS` | `--reload` | Additional flags passed to `uvicorn` |

**Example Overrides**:
```bash
make dev PORT=8080
make dev HOST=0.0.0.0 PORT=5000 UVICORN_FLAGS=""
```

---

## Target Reference

### Environment & Setup

#### `make help`
* **Purpose**: Displays a categorized list of available Makefile targets along with descriptions.
* **When to use**: Whenever you need a quick reminder of project CLI commands.
* **Command executed**: Formatted terminal output script.
* **Example usage**: `make help`

#### `make env`
* **Purpose**: Creates a local `.env` configuration file from `.env.example` if `.env` does not already exist.
* **When to use**: During initial repository setup on a new developer machine.
* **Command executed**: `cp .env.example .env` (guarded by file existence check).
* **Example usage**: `make env`

#### `make sync`
* **Purpose**: Synchronizes local virtual environment dependencies with `pyproject.toml` and `uv.lock`.
* **When to use**: After pulling updates or changing branches.
* **Command executed**: `uv sync`
* **Example usage**: `make sync`

#### `make install`
* **Purpose**: Alias for `make sync`.
* **When to use**: Standard alias for environment setup.
* **Command executed**: `make sync`
* **Example usage**: `make install`

---

### Development

#### `make dev`
* **Purpose**: Launches the FastAPI development server using Uvicorn with auto-reload.
* **When to use**: During active local development and manual API testing.
* **Command executed**: `uv run uvicorn $(APP) $(UVICORN_FLAGS) --host $(HOST) --port $(PORT)`
* **Example usage**: `make dev`

#### `make run`
* **Purpose**: Convenient alias for `make dev`.
* **When to use**: Standard shortcut for launching the development server.
* **Command executed**: `make dev`
* **Example usage**: `make run`

#### `make shell`
* **Purpose**: Launches an interactive Python REPL with project dependencies loaded.
* **When to use**: Quick interactive code testing, DB inspections, or script debugging.
* **Command executed**: `uv run python`
* **Example usage**: `make shell`

---

### Quality & Formatting

#### `make format`
* **Purpose**: Formats all Python files in the repository using Ruff formatter.
* **When to use**: Before committing code to maintain consistent code formatting.
* **Command executed**: `uv run ruff format .`
* **Example usage**: `make format`

#### `make format-check`
* **Purpose**: Verifies that all Python files adhere to Ruff formatting rules without editing files.
* **When to use**: In CI pipelines or automated check scripts.
* **Command executed**: `uv run ruff format --check .`
* **Example usage**: `make format-check`

#### `make lint`
* **Purpose**: Checks codebase for code quality issues and style violations using Ruff.
* **When to use**: Local checks before committing changes.
* **Command executed**: `uv run ruff check .`
* **Example usage**: `make lint`

#### `make lint-fix`
* **Purpose**: Automatically fixes all fixable linting issues identified by Ruff.
* **When to use**: When `make lint` reports fixable warnings.
* **Command executed**: `uv run ruff check --fix .`
* **Example usage**: `make lint-fix`

#### `make typecheck`
* **Purpose**: Performs static type checking across the codebase using BasedPyright.
* **When to use**: Verifying type safety across models, services, and repositories.
* **Command executed**: `uv run basedpyright`
* **Example usage**: `make typecheck`

#### `make check`
* **Purpose**: Aggregates `format-check`, `lint`, and `typecheck` into a single fail-fast verification target.
* **When to use**: Pre-flight code quality check before committing or opening a PR.
* **Command executed**: `make format-check && make lint && make typecheck`
* **Example usage**: `make check`

---

### Testing & Verification

#### `make test`
* **Purpose**: Runs the complete test suite using Pytest.
* **When to use**: Before pushing code or after modifying project logic.
* **Command executed**: `uv run pytest`
* **Example usage**: `make test`

#### `make test-unit`
* **Purpose**: Executes tests restricted to the `tests/` directory.
* **When to use**: Fast feedback during test-driven development.
* **Command executed**: `uv run pytest tests/`
* **Example usage**: `make test-unit`

#### `make test-cov`
* **Purpose**: Runs the Pytest suite with line-by-line terminal coverage reporting (`pytest-cov`).
* **When to use**: Assessing test coverage metrics across application modules.
* **Command executed**: `uv run pytest --cov=app --cov-report=term-missing`
* **Example usage**: `make test-cov`

#### `make verify`
* **Purpose**: Local pre-commit verification target running code quality checks and the full test suite.
* **When to use**: Developer sanity check before creating a Git commit or pushing code.
* **Command executed**: `make check && make test`
* **Example usage**: `make verify`

#### `make ci`
* **Purpose**: Standardized target executed by Continuous Integration runners (e.g. GitHub Actions).
* **When to use**: Called in automated CI pipeline jobs.
* **Command executed**: `make check && make test`
* **Example usage**: `make ci`

#### `make precommit`
* **Purpose**: Executes all configured `pre-commit` hooks across all repository files.
* **When to use**: Manually triggering pre-commit validation.
* **Command executed**: `uv run pre-commit run --all-files`
* **Example usage**: `make precommit`

---

### Database & Migrations

#### `make migrate`
* **Purpose**: Applies all pending Alembic migrations (`alembic upgrade head`).
* **When to use**: After pulling new database migrations or creating a schema revision.
* **Command executed**: `uv run alembic upgrade head`
* **Example usage**: `make migrate`

#### `make makemigrations`
* **Purpose**: Generates an autogenerated Alembic revision based on model changes.
* **When to use**: After modifying SQLAlchemy database models.
* **Command executed**: `uv run alembic revision --autogenerate -m "$(m)"`
* **Example usage**: `make makemigrations m="add user preferences table"`

#### `make downgrade`
* **Purpose**: Rollback database schema by one revision step.
* **When to use**: Testing migration rollbacks locally.
* **Command executed**: `uv run alembic downgrade -1`
* **Example usage**: `make downgrade`

#### `make current`
* **Purpose**: Displays the active migration revision currently applied to the target database.
* **When to use**: Inspecting database migration status.
* **Command executed**: `uv run alembic current`
* **Example usage**: `make current`

#### `make history`
* **Purpose**: Displays the complete migration revision timeline.
* **When to use**: Viewing revision history and parent-child migration branches.
* **Command executed**: `uv run alembic history`
* **Example usage**: `make history`

---

### Docker Infrastructure

#### `make docker-up`
* **Purpose**: Starts local database infrastructure. Automatically detects `docker-compose.yml` / `compose.yaml` if present, or falls back to starting the standalone `pg-local` PostgreSQL container.
* **When to use**: Starting local database dependencies before running the application.
* **Command executed**: `docker compose up -d` or `docker start pg-local`
* **Example usage**: `make docker-up`

#### `make docker-down`
* **Purpose**: Stops local database infrastructure.
* **When to use**: Shutting down local database containers.
* **Command executed**: `docker compose down` or `docker stop pg-local`
* **Example usage**: `make docker-down`

#### `make docker-logs`
* **Purpose**: Streams live output logs from database containers.
* **When to use**: Debugging database connection or query issues.
* **Command executed**: `docker compose logs -f` or `docker logs -f pg-local`
* **Example usage**: `make docker-logs`

---

### Cleanup

#### `make clean-pyc`
* **Purpose**: Removes Python bytecode files (`.pyc`, `.pyo`) and `__pycache__` folders.
* **When to use**: Fixing stale bytecode or module caching issues.
* **Command executed**: `find . -type d -name "__pycache__" -exec rm -rf {} +`
* **Example usage**: `make clean-pyc`

#### `make clean-cache`
* **Purpose**: Cleans cache directories (`.pytest_cache`, `.ruff_cache`, `.coverage`, `htmlcov`).
* **When to use**: Resetting test runner and linter cache state.
* **Command executed**: `rm -rf .pytest_cache .ruff_cache .coverage htmlcov`
* **Example usage**: `make clean-cache`

#### `make clean`
* **Purpose**: Runs `clean-pyc` and `clean-cache`.
* **When to use**: Full repository cleanup.
* **Command executed**: `make clean-pyc && make clean-cache`
* **Example usage**: `make clean`
