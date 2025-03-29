# Hybrid Linearization Framework

A framework for multi-modal classification using hybrid linearization pathways. This framework combines feature extraction, linearization techniques, and SVM optimization for improved classification performance.

## Features

- MNIST dataset processing and feature extraction
- Taylor series linearization of feature spaces
- Quality metrics computation for linearization evaluation
- SVM model optimization with multiple kernel support
- Cross-modal enhancement capabilities
- Comprehensive benchmarking and visualization tools

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd hybrid-linearization
```

2. Install dependencies:
```bash
pip install numpy torch torchvision scikit-learn scikit-image opencv-python matplotlib seaborn transformers
```

## Usage

Run the main script to demonstrate the framework:

```bash
python -m hybrid_linearization.main
```

The script will:
1. Load and preprocess the MNIST dataset
2. Extract image features
3. Perform feature space linearization
4. Train and evaluate an SVM classifier
5. Generate visualizations and benchmark reports

Results will be saved in the `output/` directory.

## Project Structure

```
hybrid_linearization/
├── data/               # Dataset storage
├── features/          # Feature extraction modules
├── linearization/     # Linearization algorithms
├── models/           # ML models and optimizers
├── evaluation/       # Benchmarking tools
└── output/           # Results and visualizations
```

## License

MIT License 