# Sequence Evolution with Evo2

This project implements DNA sequence evolution using the Evo2 language model. It uses simulated annealing to optimize DNA sequences based on the model's scoring.

## Requirements

- Python 3.11.7
- Poetry for dependency management
- CUDA-capable system (for Evo2)

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

3. Install Evo2:
```bash
git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
cd evo2
pip install .
```

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
