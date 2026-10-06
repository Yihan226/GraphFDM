# Polymer data contract

The polymer benchmark is not redistributed. Place the following assets before running `run_experiment.sh polymer`:

```text
data/
├── evaluator/
│   ├── O2.joblib
│   ├── N2.joblib
│   └── CO2.joblib
└── polymer/
    └── raw/
        ├── pretrain.csv.gz
        └── O2_N2_CO2.csv.gz

assets/tokenizers/polymer/
├── pretrain-token.node
├── pretrain-token.edge
├── pretrain-token.motif
└── pretrain-token.ring
```

`pretrain.csv.gz` requires a `smiles` column. `O2_N2_CO2.csv.gz` requires `smiles,SA,O2,N2,CO2` and 553 rows. Polymer SMILES are repeat units with wildcard atoms at polymerization sites. The evaluator files are joblib-serialized regressors with the same interface as the supplied molecule RF evaluators.

`python scripts/prepare_polymer.py` validates these assets, constructs the deterministic 337/107/109 split, and computes train-split normalization statistics.
