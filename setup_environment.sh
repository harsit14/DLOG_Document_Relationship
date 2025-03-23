# Create a virtual environment
python -m venv dlog
source dlog\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio
pip install torch-geometric
pip install networkx matplotlib scikit-learn
pip install transformers
pip install pandas numpy

# Create project directory structure
mkdir -p doc_gnn/data
mkdir -p doc_gnn/models
mkdir -p doc_gnn/utils
mkdir -p doc_gnn/scripts
mkdir -p doc_gnn/notebooks
mkdir -p doc_gnn/results
