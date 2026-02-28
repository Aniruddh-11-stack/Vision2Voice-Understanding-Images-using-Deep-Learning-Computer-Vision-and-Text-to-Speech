# ============================================================
# Vision2Voice — Developer Makefile
# ============================================================
.PHONY: help install install-dev run test lint format docker-build docker-up clean

PYTHON  ?= python3
PIP     ?= pip
PORT    ?= 8501

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:        ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev:    ## Install all dependencies including dev/test tools
	$(PIP) install -r requirements-dev.txt
	pre-commit install

run:            ## Launch the Streamlit dashboard
	PYTHONPATH=src streamlit run app/streamlit_app.py --server.port=$(PORT)

test:           ## Run unit tests with coverage report
	PYTHONPATH=src pytest tests/unit -v --cov=src/vision2voice --cov-report=term-missing

test-all:       ## Run all tests (unit + integration)
	PYTHONPATH=src pytest tests/ -v

lint:           ## Run flake8 linter
	flake8 src/ app/ tests/ --config=.flake8

format:         ## Auto-format code with black and isort
	black src/ app/ tests/
	isort src/ app/ tests/

type-check:     ## Run mypy static type checker
	mypy src/ --ignore-missing-imports

docker-build:   ## Build the Docker image
	docker build -t vision2voice:latest .

docker-up:      ## Start the app via docker-compose
	docker-compose up --build

docker-down:    ## Stop and remove docker-compose containers
	docker-compose down

clean:          ## Remove generated artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage coverage.xml
	rm -f outputs/temp_upload.jpg outputs/output_speech.mp3
