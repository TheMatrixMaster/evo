# Quick Usage Guide

## Installation

Dependencies are already in `pyproject.toml`. Model weights are automatically loaded.

## Basic Usage

```python
from evo.oracles import get_oracle

# Load any oracle
oracle = get_oracle("clone")                              # LLM oracle
oracle = get_oracle("rand_0.5")                           # Random oracle (alpha=0.5)
oracle = get_oracle("SARSCoV1", device="cpu")             # COVID oracle (CPU mode)
oracle = get_oracle("SARSCoV2Beta", device="cpu")         # COVID oracle (CPU mode)

# Predict fitness
fitness = oracle("QITLKESGPTLVKPTQTLTLTCTFSGFSLNTRGVSVAWIRQPPGKALD...")
fitnesses = oracle([seq1, seq2, seq3])  # Batch prediction

# Access precomputed seed fitnesses (instant, no inference)
seed_fitness = oracle.get_seed_fitness("seed_0")
all_fitnesses = oracle.seed_fitnesses
```

## Available Oracles

| Name | Description | Device | Higher is Better? |
|------|-------------|--------|-------------------|
| `clone` | General antibody fitness | GPU | No |
| `rand_X` | Fitness with noise (X = alpha 0.0-1.0) | GPU | No |
| `SARSCoV1` | SARS-CoV-1 neutralization | **CPU** | Yes |
| `SARSCoV2Beta` | SARS-CoV-2 Beta neutralization | **CPU** | Yes |

**Important:** COVID oracles must use `device="cpu"`.

## Example Script

Run the included example:

```bash
python evo/evo/oracles/example.py
```

## Key Features

✅ Unified interface for all oracles  
✅ Precomputed seed fitnesses (no inference needed)  
✅ Batch prediction support  
✅ Automatic chain type detection  
✅ No PyTorch version changes required  

See `README.md` for detailed documentation.
