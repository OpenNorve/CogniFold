#!/bin/bash
# Pre-commit hook for Cognifold
# Run this before committing to ensure code quality
# To install as git hook: cp scripts/pre-commit.sh .git/hooks/pre-commit

set -e

echo "Running pre-commit checks..."

# Format check
echo "Checking formatting..."
python -m ruff format --check src/ tests/

# Lint check
echo "Checking linting..."
python -m ruff check src/ tests/

# Run tests
echo "Running tests..."
python -m pytest tests/ -q

echo "All checks passed!"
