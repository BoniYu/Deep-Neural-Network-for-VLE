# Deep Neural Network for VLE Prediction

Predicting the vapor pressure of pure compounds (the vapor-liquid equilibrium curve) with a deep neural network built in PyTorch, using molecular fingerprints and temperature as inputs.

Based on tutorial 4 of the [Machine Learning in Chemical Engineering](https://edgarsmdn.github.io/MLCE_book/04_DNN_VLE.html) book (Sanchez Medina, del Rio Chanona, Ganzer), with my own extensions.

## Goals

- Understand why mechanistic knowledge (Clausius-Clapeyron) matters when building ML models for chemical engineering
- Learn to build and train deep neural networks with PyTorch
- Predict vapor pressures for compounds not seen during training (extrapolation in chemical space)

## Data

| File | Description |
|------|-------------|
| `data/CO2_data.csv` | CO2 saturation data (T, P_vap, H_vap, rho_vap, rho_liq) |
| `data/Vapor_pressures.csv` | Vapor pressures of many compounds with names, SMILES, chemical class and subclass |

Both files come from the [MLCE_book repository](https://github.com/edgarsmdn/MLCE_book/tree/main/references).

## Project structure

```
.
├── README.md
├── REPORT.md
├── LICENSE
├── requirements.txt
├── data/
└── notebooks/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Status

In progress. See [REPORT.md](REPORT.md) for the running log of what has been done and learned.

## License

MIT, see [LICENSE](LICENSE).
