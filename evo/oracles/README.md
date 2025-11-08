# Protein Fitness Oracles

Unified interface for protein fitness prediction using models from the CloneBO framework. All 4 oracles are fully functional with precomputed seed sequences and fitnesses.

## Quick Start

```python
from evo.oracles import get_oracle

# LLM Oracle - language model-based fitness
oracle = get_oracle("clone")
fitness = oracle("QITLKESGPTLVKPTQTLTLTCTFSGFSLNTRGVSVAWIRQPPGKALD...")
print(f"Fitness: {fitness} (lower is better)")

# COVID Oracles - neutralization prediction (use CPU mode)
oracle = get_oracle("SARSCoV1", device="cpu")
oracle = get_oracle("SARSCoV2Beta", device="cpu")

# Random Oracle - hybrid LLM + noise
oracle = get_oracle("rand_0.5")  # alpha=0.5 (balanced)
oracle = get_oracle("rand_0.0")  # pure LLM
oracle = get_oracle("rand_1.0")  # pure random
```

## Available Oracles

| Oracle | Output | Higher is Better? | Device | Use Case |
|--------|--------|-------------------|--------|----------|
| `clone` | Cross-entropy loss | ❌ No | GPU | General antibody fitness |
| `SARSCoV1` | Neutralization logits | ✅ Yes | **CPU** | SARS-CoV-1 binding |
| `SARSCoV2Beta` | Neutralization logits | ✅ Yes | **CPU** | SARS-CoV-2 Beta binding |
| `rand_X` | Normalized loss | ❌ No | GPU | Fitness with controllable noise |

**Note:** COVID oracles must use `device="cpu"` due to SRU CUDA compilation issues with PyTorch 2.6+. The CPU fallback works perfectly!

## Precomputed Seed Sequences

All oracles include precomputed seed sequences and fitnesses from CloneBO:

```python
# Access seed sequences
oracle = get_oracle("clone")
sequences = oracle.seed_sequences  # List of seed sequences
fitnesses = oracle.seed_fitnesses   # List of precomputed fitnesses

# Query specific seed fitness (no inference needed!)
fitness = oracle.get_seed_fitness("seed_0")

# Access full seed data
seed_data = oracle.seed_data
# {'seed_0': {'sequence': '...', 'fitness': 0.228},
#  'seed_1': {'sequence': '...', 'fitness': 0.450}}
```

### Seed Fitness Values

| Oracle | seed_0 | seed_1 |
|--------|--------|--------|
| `clone` | 0.2285 | 0.4508 |
| `rand_0.5` | -0.7133 | -0.2916 |
| `SARSCoV1` | -0.9165 | -0.9123 |
| `SARSCoV2Beta` | 1.9333 | 0.2185 |

## Common Interface

All oracles inherit from `BaseOracle` and provide:

```python
# Single sequence prediction
fitness = oracle.predict(sequence)

# Batch prediction (faster)
fitnesses = oracle.predict_batch([seq1, seq2, seq3])

# Callable interface (accepts string or list)
fitness = oracle(sequence)
fitnesses = oracle([seq1, seq2, seq3])

# Properties
chain_type = oracle.chain_type           # "heavy", "light", or "both"
is_higher_better = oracle.higher_is_better  # True or False
device = oracle.device                    # "cuda" or "cpu"
```

## Examples

### Basic Usage

```python
from evo.oracles import get_oracle

# Load oracle
oracle = get_oracle("clone")

# Single prediction
sequence = "QITLKESGPTLVKPTQTLTLTCTFSGFSLNTRGVSVAWIRQPPGKALD..."
fitness = oracle(sequence)
print(f"Fitness: {fitness:.4f}")

# Batch prediction
sequences = [seq1, seq2, seq3]
fitnesses = oracle(sequences)
```

### Using COVID Oracles

```python
from evo.oracles import get_oracle

# Load COVID oracle (must use CPU)
oracle = get_oracle("SARSCoV1", device="cpu", use_iglm_weighting=False)

# Predict neutralization
sequence = "QVQLQESGPGLVKPSETLSLTCTVSGGFIGPHYWSWVRQPPGKGLEWI..."
neutralization = oracle(sequence)
print(f"Neutralization: {neutralization:.4f} (higher is better)")
```

### Using Random Oracle with Different Alpha

```python
from evo.oracles import get_oracle

# Different noise levels
oracle_pure_llm = get_oracle("rand_0.0")     # alpha=0: pure LLM
oracle_balanced = get_oracle("rand_0.5")     # alpha=0.5: balanced
oracle_pure_rand = get_oracle("rand_1.0")    # alpha=1: pure random

# Compare predictions
sequence = "QITLKESGPTLVKPTQTLTLTCTFSGFSLNTRGVSVAWIRQPPGKALD..."
fitness_llm = oracle_pure_llm(sequence)
fitness_balanced = oracle_balanced(sequence)
fitness_rand = oracle_pure_rand(sequence)
```

### Finding Best Sequence

```python
from evo.oracles import get_oracle
import numpy as np

oracle = get_oracle("SARSCoV1", device="cpu")
sequences = [seq1, seq2, seq3, seq4, seq5]
fitnesses = oracle(sequences)

# Find best sequence
if oracle.higher_is_better:
    best_idx = np.argmax(fitnesses)
else:
    best_idx = np.argmin(fitnesses)

print(f"Best sequence: {sequences[best_idx]}")
print(f"Best fitness: {fitnesses[best_idx]:.4f}")
```

## Installation

Dependencies are included in the main `pyproject.toml`:
- `transformers>=4.30.0` - For LLM oracle
- `sru>=2.6.0` - For COVID oracle RNN
- `abnumber>=0.3.0` - For antibody alignment
- `iglm>=0.1.0` - For COVID oracle IgLM weighting

Model weights are automatically loaded from:
`/scratch/users/stephen.lu/projects/protevo/checkpoints/oracle_weights/`

## Technical Notes

- **Chain Type:** All oracles work with heavy chain sequences only
- **COVID Oracle Device:** Use `device="cpu"` due to SRU CUDA compilation issues
- **IgLM Weighting:** COVID oracles support optional IgLM likelihood weighting (requires HuggingFace auth)
- **Random Oracle Alignment:** Internally aligns sequences using AHO numbering scheme
- **Seed Sequences:** First 2 sequences (indices 0-1) from each oracle's training dataset

## Files

```
oracles/
├── base.py           # Base oracle class
├── llm_oracle.py     # LLM-based oracle
├── covid_oracle.py   # COVID neutralization oracles
├── rand_oracle.py    # Random hybrid oracle
├── covid_model.py    # SRU-based COVID model architecture
├── bytenet.py        # ByteNet architecture for random oracle
├── __init__.py       # Factory function and exports
└── README.md         # This file
```

## Citation

If you use these oracles, please cite the CloneBO paper:

```bibtex
@article{clonebo2024,
  title={CloneBO: Bayesian Optimization for Antibody Design using Clonal Families},
  author={...},
  year={2024}
}
```
