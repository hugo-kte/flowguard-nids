PY      ?= .venv/bin/python
CONFIG  ?= configs/cicids2017.yaml
export PYTHONPATH := src

DATA_URL    := https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip
DATA_FILE   := data/raw/CICIDS2017_improved.zip
DATA_SHA256 := 97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883

.DEFAULT_GOAL := help
.PHONY: help install data train evaluate pipeline quickstart replay dashboard test lint format docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install FlowGuard with dashboard + dev extras
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dashboard,dev]"

data: $(DATA_FILE) ## Download the improved CICIDS2017 dataset (328 MB) and verify its SHA-256

$(DATA_FILE):
	mkdir -p $(dir $@)
	curl -fL --retry 3 -o $@.part $(DATA_URL)
	$(PY) -c "import hashlib,sys; h=hashlib.sha256(open('$@.part','rb').read()).hexdigest(); sys.exit(0 if h=='$(DATA_SHA256)' else 'checksum mismatch: '+h)"
	mv $@.part $@

train: ## Train every model on $(CONFIG)
	$(PY) -m flowguard train -c $(CONFIG)

evaluate: ## Evaluate on the test split, write reports/ (metrics, figures, REPORT.md)
	$(PY) -m flowguard evaluate -c $(CONFIG)

pipeline: data train evaluate ## Full CICIDS2017 run: download, train, evaluate

quickstart: ## 1-minute end-to-end run on synthetic traffic (no download needed)
	$(PY) -m flowguard train -c configs/synthetic.yaml
	$(PY) -m flowguard evaluate -c configs/synthetic.yaml
	$(PY) -m flowguard demo-pcap demo/demo.pcap
	$(PY) -m flowguard detect demo/demo.pcap -c configs/synthetic.yaml

replay: ## Stream the labelled test split through the engine (feeds the dashboard)
	$(PY) -m flowguard replay -c $(CONFIG) --rate 1500

dashboard: ## Launch the SOC dashboard on http://localhost:8501
	$(PY) -m flowguard dashboard -c $(CONFIG)

test: ## Run the test suite with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

lint: ## Lint and check formatting
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format: ## Auto-format the code
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

docker-build: ## Build the container image
	docker build -t flowguard:latest .

docker-up: ## Replay + dashboard in containers (needs trained artifacts/)
	docker compose up --build

clean: ## Remove caches and runtime output (keeps data and models)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov runtime demo
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
