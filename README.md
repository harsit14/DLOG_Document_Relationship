# Document Relationship Modeling Using Graph Neural Networks

This repository implements the "Document Relationship Modeling Using Graph Neural Networks" project proposed by Harsit Upadhya and Zeyuan Meng from Emory University.

## Project Overview

This project investigates the application of Graph Neural Networks (GNNs) to effectively model and analyze document interconnections. By integrating citation networks, semantic similarity metrics, and co-authorship frameworks, we construct sophisticated GNN-based models to enhance document classification, citation prediction, and document clustering.

### Key Features

- **Graph Neural Network Models**: Implementation of GCN, GAT, and GraphSAGE
- **Document Classification**: Classify documents based on their content and connectivity
- **Link Prediction**: Predict potential citation relationships between documents
- **Document Clustering**: Group similar documents based on their embeddings
- **Community Detection**: Identify document communities in similarity networks

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-enabled GPU (recommended but not required)

### Setup Instructions

1. **Clone the repository and navigate to the project directory**

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv dlog
   dlog\Scripts\activate  # On Windows
   # or
   source dlog/bin/activate  # On Unix/MacOS
   ```

3. **Install PyTorch** (skip if already installed):
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Install PyTorch Geometric and its dependencies**:
   ```bash
   # For CUDA 12.1 with PyTorch 2.1.0 (adjust based on your PyTorch/CUDA version)
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
   
   # For CPU only
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
   ```

5. **Install other dependencies**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tqdm networkx python-louvain
   ```

6. **Create necessary directories**:
   ```bash
   mkdir -p data results
   ```

## Project Structure

```
project_folder/
├── main_script.py             # Main script for running the full pipeline
├── data_preprocessing.py      # Data loading and preprocessing utilities
├── gnn_models.py              # GNN model implementations (GCN, GAT, GraphSAGE)
├── training_evaluation.py     # Training and evaluation procedures
├── document_clustering.py     # Document clustering functionality
├── example_classification.py  # Script for running document classification
├── example_link_prediction.py # Script for running link prediction
├── example_clustering.py      # Script for running document clustering
├── config-file.json           # Configuration file
├── readme.md                  # Project documentation
├── requirements.txt           # Package dependencies
├── setup_environment.sh       # Environment setup script
└── data/                      # Directory for datasets (CORA will be downloaded here)
```

## Usage

### Running the Full Pipeline

To run the full pipeline with default settings:

```bash
python main_script.py
```

### Running Individual Tasks

You can run individual task scripts:

```bash
# Document classification only
python example_classification.py

# Link prediction only
python example_link_prediction.py

# Document clustering only
python example_clustering.py
```

### Customizing Runs with Command-Line Arguments

```bash
# Using GAT model with custom settings
python main_script.py --model gat --hidden_dim 128 --epochs 300

# Adjusting clustering parameters
python example_clustering.py --clustering_method hierarchical --n_clusters 10
```

## Implementation Notes

- **Dynamic Graph Construction**: The project builds document graphs using citation relationships from the CORA dataset.
- **GNN Models**: All models (GCN, GAT, GraphSAGE) are implemented using PyTorch Geometric's neural network modules.
- **Evaluation Metrics**: 
  - For classification: Accuracy, precision, recall, F1 score
  - For link prediction: AUC, precision, recall, F1 score
  - For clustering: NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index)

## Troubleshooting

- **CUDA issues**: If you encounter CUDA errors, try running with `--device cpu`:
  ```bash
  python main_script.py --device cpu
  ```

- **Library installation issues**: For PyTorch Geometric, ensure the versions match your CUDA and PyTorch versions. See the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

- **Community detection errors**: Ensure you've installed the python-louvain package:
  ```bash
  pip install python-louvain
  ```

## Dataset

The implementation primarily uses the CORA dataset:

- **CORA**: A benchmark dataset containing 2,708 scientific publications classified into 7 categories and connected by 5,429 citation links.

## Citation


## License

This project is licensed under the MIT License.

## Acknowledgments

- The project is based on the research proposal by Upadhya and Meng from Emory University
- The implementation uses PyTorch and PyTorch Geometric for GNN models
