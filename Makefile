# Shortcuts for common commands. Without make (e.g. on Windows), run the uv commands directly.
CONFIG ?= configs/baseline.yaml

.PHONY: setup fmt lint test check eval regress

setup:
	uv sync

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

check: lint test

# WS4: run the dev split through CONFIG, e.g. `make eval CONFIG=configs/experiments/k3.yaml`
eval:
	uv run deskrag eval run --config $(CONFIG) --split dev

# WS5: requires a promoted baseline and WS4's eval runner.
BASELINE ?= baselines/dev
GATES ?= configs/regression.yaml
regress:
	uv run deskrag regress --config $(CONFIG) --baseline $(BASELINE) --gates $(GATES)
