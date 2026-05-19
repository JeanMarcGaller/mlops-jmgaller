.PHONY: help sync test coverage coverage-html lint format quality hooks weather hopsworks feature-dry feature compare-models train train-cache train-local tune tune-fast tune-cache tune-fast-cache select select-cache final final-cache final-local backtest infer monitor dashboard e2e e2e-local docker-build docker-infer docker-feature-dry

help:
	@echo "Available commands:"
	@echo "  make sync               Install dependencies with uv"
	@echo "  make test               Run pytest"
	@echo "  make coverage           Run pytest with terminal coverage report"
	@echo "  make coverage-html      Run pytest with terminal and HTML coverage reports"
	@echo "  make lint               Run Ruff lint check"
	@echo "  make format             Format code with Ruff"
	@echo "  make quality            Run Ruff lint, Ruff format check, and pytest"
	@echo "  make hooks              Run pre-commit hooks on all files"
	@echo "  make weather            Run Open-Meteo API smoke test"
	@echo "  make hopsworks          Run Hopsworks connection smoke test"
	@echo "  make feature-dry        Build features locally without Hopsworks write"
	@echo "  make feature            Build features and write to Hopsworks"
	@echo "  make compare-models     Compare HistGBR, Random Forest and XGBoost"
	@echo "  make train              Run baseline training pipeline"
	@echo "  make train-cache        Run baseline training with local feature cache"
	@echo "  make train-local        Run baseline training with cache and skip registry upload"
	@echo "  make tune-fast          Run Optuna tuning with 5 trials"
	@echo "  make tune-fast-cache    Run Optuna tuning with 5 trials from local cache"
	@echo "  make tune               Run Optuna tuning with 30 trials"
	@echo "  make tune-cache         Run Optuna tuning with 30 trials from local cache"
	@echo "  make select             Run feature selection pipeline"
	@echo "  make select-cache       Run feature selection from local cache"
	@echo "  make final              Run final training pipeline"
	@echo "  make final-cache        Run final training with local feature cache"
	@echo "  make final-local        Run final training with cache and skip registry upload"
	@echo "  make backtest           Run local expanding walk-forward backtesting"
	@echo "  make infer              Run inference pipeline"
	@echo "  make monitor            Run monitoring pipeline for logged predictions"
	@echo "  make dashboard          Run local Streamlit dashboard"
	@echo "  make e2e                Run full end-to-end workflow with Hopsworks"
	@echo "  make e2e-local          Run local experiment workflow with cache and no model upload"
	@echo "  make docker-build       Build Docker image for linux/amd64"
	@echo "  make docker-infer       Run inference pipeline in Docker"
	@echo "  make docker-feature-dry Run feature dry-run in Docker"

sync:
	uv sync

test:
	uv run pytest

coverage:
	uv run pytest --cov=src --cov-report=term-missing

coverage-html:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check .

format:
	uv run ruff format .

quality:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

hooks:
	uv run pre-commit run --all-files

weather:
	uv run python src/weather_api.py

hopsworks:
	uv run python src/hopsworks_client.py

feature-dry:
	uv run python src/feature_pipeline.py --days 30 --dry-run

feature:
	uv run python src/feature_pipeline.py --days 365

compare-models:
	uv run python src/model_comparison_pipeline.py --use-cache

train:
	uv run python src/baseline_training_pipeline.py

train-cache:
	uv run python src/baseline_training_pipeline.py --use-cache

train-local:
	uv run python src/baseline_training_pipeline.py --use-cache --skip-upload

tune-fast:
	uv run python src/tuning_pipeline.py --trials 5

tune-fast-cache:
	uv run python src/tuning_pipeline.py --trials 5 --use-cache

tune:
	uv run python src/tuning_pipeline.py --trials 30

tune-cache:
	uv run python src/tuning_pipeline.py --trials 30 --use-cache

select:
	uv run python src/feature_selection_pipeline.py

select-cache:
	uv run python src/feature_selection_pipeline.py --use-cache

final:
	uv run python src/final_training_pipeline.py

final-cache:
	uv run python src/final_training_pipeline.py --use-cache

final-local:
	uv run python src/final_training_pipeline.py --use-cache --skip-upload

backtest:
	uv run python src/backtesting_pipeline.py

infer:
	uv run python src/inference_pipeline.py

monitor:
	uv run python src/monitoring_pipeline.py

dashboard:
	uv run python -m streamlit run dashboard/app.py

e2e:
	uv run python src/feature_pipeline.py --days 365
	uv run python src/baseline_training_pipeline.py
	uv run python src/tuning_pipeline.py --trials 30
	uv run python src/feature_selection_pipeline.py
	uv run python src/final_training_pipeline.py
	uv run python src/inference_pipeline.py

e2e-local:
	uv run python src/tuning_pipeline.py --trials 5 --use-cache
	uv run python src/feature_selection_pipeline.py --use-cache
	uv run python src/final_training_pipeline.py --use-cache --skip-upload
	uv run python src/model_comparison_pipeline.py --use-cache
	uv run python src/backtesting_pipeline.py

docker-build:
	docker build --platform linux/amd64 -t mlops-temperature-forecasting .

docker-infer:
	docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting

docker-feature-dry:
	docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting \
		uv run python src/feature_pipeline.py --days 30 --dry-run