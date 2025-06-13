# LLM Fine-tuning Project

This project implements fine-tuning of Large Language Models (LLMs) using various approaches including full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and QLoRA.

## Project Structure

```
.
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed datasets
│   └── synthetic/         # Synthetically generated data
├── src/                   # Source code
│   ├── data/             # Data processing scripts
│   ├── models/           # Model definitions and training code
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks for experimentation
├── tests/               # Unit tests
└── scripts/             # Training and evaluation scripts
```

## Setup

1. Create a virtual environment:
```bash
python -m venv llm_finetune
source llm_finetune/bin/activate  # Linux/Mac
# or
llm_finetune\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place your raw data in `data/raw/`
2. Run data preprocessing:
```bash
python src/data/preprocess.py
```

### Training
1. Configure training parameters in `configs/training_config.yaml`
2. Start training:
```bash
python scripts/train.py
```

### Evaluation
```bash
python scripts/evaluate.py
```

## Features

- Support for multiple fine-tuning approaches:
  - Full fine-tuning
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
- Data preprocessing and augmentation
- Comprehensive evaluation metrics
- Experiment tracking with Weights & Biases
- Model deployment utilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 