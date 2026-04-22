.PHONY: help setup install train test run clean clean-all

CONDA_ENV := pspi
PYTHON_VERSION := 3.11.5

help:
	@echo "PSPI: Prokaryote SP Identifier - Available targets:"
	@echo ""
	@echo "  make setup          - Create conda environment (Python 3.11.5) and install dependencies"
	@echo "  make install        - Install dependencies in conda environment"
	@echo "  make train MODEL=name.pkl - Train a new model (saves as name.pkl)"
	@echo "  make test MODEL=name.pkl  - Test an existing model on datasets/testing"
	@echo "  make run             - Run inference on files in input_files/"
	@echo "  make run-file [FILE=path] INPUT=path OUTPUT=path [THRESHOLD=0.75] - Run on specific file (FILE defaults to model_default.pkl)"
	@echo "  make clean           - Remove output files"
	@echo "  make clean-all       - Remove output files and conda environment"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make train MODEL=my_model.pkl"
	@echo "  make test MODEL=model_default.pkl"
	@echo "  make run-file INPUT=test.fasta OUTPUT=results.csv"
	@echo "  make run-file FILE=my_model.pkl INPUT=test.fasta OUTPUT=results.csv THRESHOLD=0.35"

setup:
	@echo "Setting up PSPI environment with Python $(PYTHON_VERSION)..."
	conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	conda run -n $(CONDA_ENV) pip install --upgrade pip setuptools wheel
	conda run -n $(CONDA_ENV) pip install tensorflow==2.15.0 keras==2.15.0 numpy==1.24.3 scikit-learn==1.3.0 jupyter ipykernel
	@echo "Setup complete! Activate the environment with: conda activate $(CONDA_ENV)"

install:
	@echo "Installing PSPI dependencies..."
	conda run -n $(CONDA_ENV) pip install --upgrade pip setuptools wheel
	conda run -n $(CONDA_ENV) pip install tensorflow==2.15.0 keras==2.15.0 numpy==1.24.3 scikit-learn==1.3.0 jupyter ipykernel
	@echo "Installation complete!"

train:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make train MODEL=my_model.pkl"; \
		exit 1; \
	fi
	@echo "Training model: $(MODEL)"
	conda run -n $(CONDA_ENV) python pspi.py -t $(MODEL)

test:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make test MODEL=my_model.pkl"; \
		exit 1; \
	fi
	@echo "Testing model: $(MODEL)"
	conda run -n $(CONDA_ENV) python pspi.py -r $(MODEL)

run:
	@echo "Running inference on input_files/..."
	conda run -n $(CONDA_ENV) python pspi.py

run-file:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: INPUT and OUTPUT parameters required"; \
		echo "Usage: make run-file [FILE=model.pkl] INPUT=input.fasta OUTPUT=output.csv [THRESHOLD=0.75]"; \
		exit 1; \
	fi
	@MODEL_PARAM=$${FILE:-model_default.pkl}; \
	if [ -z "$(THRESHOLD)" ]; then \
		echo "Running inference with model: $$MODEL_PARAM, input: $(INPUT), output: $(OUTPUT)"; \
		conda run -n $(CONDA_ENV) python pspi.py -m $$MODEL_PARAM -i $(INPUT) -o $(OUTPUT); \
	else \
		echo "Running inference with model: $$MODEL_PARAM, input: $(INPUT), output: $(OUTPUT), threshold: $(THRESHOLD)"; \
		conda run -n $(CONDA_ENV) python pspi.py -m $$MODEL_PARAM -i $(INPUT) -o $(OUTPUT) -l $(THRESHOLD); \
	fi

clean:
	@echo "Cleaning output files..."
	rm -f output_files/results.csv
	rm -f output_files/*.csv
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing conda environment: $(CONDA_ENV)..."
	conda env remove -n $(CONDA_ENV) -y
	@echo "Clean complete!"
