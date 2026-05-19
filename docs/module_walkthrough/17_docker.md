# Docker

## Role in the Project

The Docker setup allows the project to run inside a containerized Python environment.

The default container command runs the live inference pipeline:

```bash
uv run python src/inference_pipeline.py
```

This makes the inference workflow portable and independent of the local Python environment.

## Inputs

The Docker image includes:

* `pyproject.toml`
* `uv.lock`
* `src/`
* `README.md`
* `.env.example`

The real `.env` file is not copied into the image. It is passed at runtime with:

```bash
docker run --env-file .env ...
```

This keeps secrets such as the Hopsworks API key out of the Docker image.

## Outputs

The default container run executes inference.

Depending on the command and mounted volumes, the pipeline can produce local artifacts such as:

```text
reports/latest_prediction.json
reports/prediction_log.csv
```

However, because the current Docker command does not mount a host volume, files written inside the container are removed when the container exits.

For persistent local reports, the project is usually run directly on the host or the Docker command would need a volume mount.

## Dockerfile Structure

### Base Image

The image uses:

```dockerfile
FROM python:3.12-slim
```

This provides a small Python 3.12 Linux environment.

### Working Directory

The container working directory is:

```dockerfile
WORKDIR /app
```

### Environment Variables

The Dockerfile sets:

```dockerfile
PYTHONUNBUFFERED=1
PYTHONPATH=/app/src
```

`PYTHONUNBUFFERED=1` makes logs appear immediately in the terminal.

`PYTHONPATH=/app/src` allows modules inside `src/` to be imported directly.

### System Dependencies

The image installs basic build tools and curl:

```dockerfile
build-essential
curl
```

These are needed for some Python packages that may require compilation or downloads during installation.

### Dependency Installation

The Dockerfile copies only dependency files first:

```dockerfile
COPY pyproject.toml uv.lock ./
```

Then it installs `uv` and syncs dependencies:

```dockerfile
pip install --no-cache-dir uv
uv sync --frozen --no-dev
```

`--frozen` ensures the lockfile is respected.

`--no-dev` excludes development dependencies such as testing and linting tools.

### Project Files

The image copies:

```dockerfile
COPY src ./src
COPY README.md ./
COPY .env.example ./
```

The real `.env` file is intentionally not copied.

### Default Command

The default command is:

```dockerfile
CMD ["uv", "run", "python", "src/inference_pipeline.py"]
```

So running the container without overriding the command runs inference.

## `.dockerignore`

The `.dockerignore` file excludes local-only, generated or sensitive files from the Docker build context.

Excluded files and directories include:

```text
.env
.venv
.git
.github
__pycache__
.pytest_cache
.ruff_cache
data/raw
data/features
models
reports
input_example.json
*.zip
```

This keeps the Docker image smaller and prevents secrets or generated artifacts from being copied into the image.

## Runtime Configuration

The container needs a `.env` file at runtime for Hopsworks credentials and project settings.

Example:

```bash
docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting
```

The `.env` file is passed into the container environment but is not baked into the image.

## Makefile Commands

### `make docker-build`

Builds the Docker image:

```bash
docker build --platform linux/amd64 -t mlops-temperature-forecasting .
```

### `make docker-infer`

Runs the default inference pipeline inside Docker:

```bash
docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting
```

### `make docker-feature-dry`

Runs a feature pipeline dry-run inside Docker:

```bash
docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting \
  uv run python src/feature_pipeline.py --days 30 --dry-run
```

## Key Design Decisions

### Secrets Are Not Copied

The real `.env` file is excluded through `.dockerignore`.

Secrets are provided only at runtime with `--env-file .env`.

### Default Command Is Inference

The Docker image defaults to running the inference pipeline.

This makes the container useful as a simple serving-style batch inference image.

### `linux/amd64` Platform

The Makefile builds and runs the image with:

```text
linux/amd64
```

This is useful on Apple Silicon Macs because some Hopsworks-related dependencies may not provide Linux ARM64 wheels.

### Development Files Are Excluded

Tests, local caches, reports, models and virtual environments are not copied into the Docker image.

The image is therefore focused on running the application pipeline, not on local development.

### Dependency Lockfile Is Used

The Docker build uses `uv.lock` with `uv sync --frozen`.

This improves reproducibility because dependency versions come from the lockfile.

## README-Relevant Points

* The project includes a Dockerfile for containerized execution.
* The default Docker command runs the inference pipeline.
* The real `.env` file is not copied into the image.
* Runtime secrets are passed with `--env-file .env`.
* The Makefile builds and runs Docker with `linux/amd64`.
* Local artifacts such as reports and models are excluded from the Docker build context.
* Docker is mainly used for reproducible inference or pipeline execution, not for local report persistence.

## Limitations / Future Work

* The default Docker run does not mount local volumes, so generated reports inside the container are not persisted after the container exits.
* The image currently copies only `src/`, not tests or dashboard files.
* The default command is inference-focused.
* A future Docker setup could add separate images or commands for dashboard, training and scheduled inference.
* A future version could mount `reports/` and `models/` as volumes for persistent artifacts.
* A production deployment would likely use a scheduler or orchestrator instead of manually running `docker run`.
