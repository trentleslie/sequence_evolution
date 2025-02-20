# Sequence Evolution with Evo2

This project implements DNA sequence evolution using the Evo2 language model. It uses simulated annealing to optimize DNA sequences based on the model's scoring.

## Requirements

- Python 3.11.7
- Poetry for dependency management
- CUDA-capable system with:
  - At least 10GB VRAM for `evo2_7b` (due to memory overhead and fragmentation)
  - At least 2GB VRAM for `evo2_1b_base`
  - Multiple GPUs for `evo2_40b`
- CUDA toolkit and development libraries

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd sequence_evolution
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Install CUDA development libraries (if not already installed):
```bash
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev
```

4. Install required Python packages:
```bash
poetry run pip install flash-attn==2.6.3  # Specific version required for compatibility
poetry run pip install vortex  # Required for Evo2
transformer-engine  # Required for model operations
```

5. Install Evo2:
```bash
git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
cd evo2
pip install .
```

## Model Selection

Evo2 provides several model sizes:
- `evo2_40b`: Requires multiple GPUs
- `evo2_7b`: Requires ~10GB VRAM (actual model is ~7GB but needs extra memory for operations)
- `evo2_1b_base`: Smallest model, requires ~2GB VRAM

Choose the appropriate model based on your available GPU memory.

## Troubleshooting

### CUDA Out of Memory
If you encounter CUDA out of memory errors:
1. Try using a smaller model (e.g., `evo2_1b_base` instead of `evo2_7b`)
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid memory fragmentation
3. Use CPU inference by setting `CUDA_VISIBLE_DEVICES=""` (will be slower)

### Flash Attention Version
If you see warnings about flash-attn version compatibility:
1. Uninstall current flash-attn
2. Install version 2.6.3 specifically: `pip install flash-attn==2.6.3`

### Missing CUDA Libraries
If you encounter missing CUDA library errors:
1. Install CUDA toolkit
2. Install cuDNN libraries: `sudo apt-get install libcudnn8 libcudnn8-dev`

## Project Structure

```
sequence_evolution/
├── pyproject.toml        # Project dependencies and configuration
├── README.md            # Project documentation
├── src/                 # Source code
│   └── sequence_evolution/
│       ├── __init__.py
│       └── evolver.py   # Main sequence evolution implementation
└── tests/              # Test files
    └── test_evolver.py
```

## Usage

```python
from sequence_evolution.evolver import SequenceEvolver
from evo2 import Evo2

# Initialize Evo2 model
model = Evo2('evo2_7b')

# Create sequence evolver
evolver = SequenceEvolver(model)

# Evolve a sequence
evolved_sequence, history = evolver.evolve_sequence()
```

## License

[Your chosen license]
