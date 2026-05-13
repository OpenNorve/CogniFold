# Cognifold Development Makefile
# ==============================
# Common development tasks for the Cognifold project.
# Usage: make <target>

.PHONY: all install dev test lint format check clean help

# Default target
all: check test

# Install dependencies
install:
	pip install -e .

# Install with development dependencies
dev:
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
coverage:
	python -m pytest tests/ -v --cov=cognifold --cov-report=html --cov-report=term

# Run linting
lint:
	python -m ruff check src/ tests/

# Run formatter
format:
	python -m ruff format src/ tests/

# Fix linting issues
fix:
	python -m ruff check src/ tests/ --fix
	python -m ruff format src/ tests/

# Run type checking
typecheck:
	python -m pyright src/

# Run all quality checks (lint + format check + type check)
check: lint
	python -m ruff format --check src/ tests/

# Run full quality gate (all checks + tests)
quality: check test

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Generate events
generate-personal:
	python -m cognifold.cli generate --domain personal-timeline --persona software_engineer --events 50

generate-computer:
	python -m cognifold.cli generate --domain computer-activity --profile software_developer --events 50

generate-service:
	python -m cognifold.cli generate --domain service-logs --topology ecommerce --events 50

# Build wiki timeline
build-wiki:
	python -m cognifold.cli build-timeline --source wiki --input data/wiki/ -o data/generated/wiki_timeline.json

# Run simulation
run:
	python -m cognifold.cli run data/mock_timeline.json --agent -o output/

# Show help
help:
	@echo "Cognifold Development Tasks"
	@echo "==========================="
	@echo ""
	@echo "Setup:"
	@echo "  install     Install package"
	@echo "  dev         Install with dev dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  lint        Run linting"
	@echo "  format      Run formatter"
	@echo "  fix         Fix linting issues and format"
	@echo "  typecheck   Run type checking"
	@echo "  check       Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run tests"
	@echo "  coverage    Run tests with coverage"
	@echo "  quality     Run quality + tests"
	@echo ""
	@echo "Generation:"
	@echo "  generate-personal   Generate personal timeline events"
	@echo "  generate-computer   Generate computer activity events"
	@echo "  generate-service    Generate service log events"
	@echo "  build-wiki          Build wiki timeline from files"
	@echo ""
	@echo "Simulation:"
	@echo "  run         Run simulation with agent"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean       Clean build artifacts"
	@echo "  help        Show this help"
