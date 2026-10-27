# Deep Neural Network for VLE Prediction

Predicting the vapor pressure of pure compounds (the vapor-liquid equilibrium curve) from molecular structure and temperature, and comparing a PyTorch deep neural network with tree-based models (Decision Tree, Random Forest, XGBoost).

Based on tutorial 4 of the [Machine Learning in Chemical Engineering](https://edgarsmdn.github.io/MLCE_book/04_DNN_VLE.html) book (Sanchez Medina, del Rio Chanona, Ganzer), with my own extensions.

## Highlights

- **Task:** predict $\ln P_{vap}$ from a 2048-bit molecular fingerprint (made from the SMILES string) plus temperature, for **compounds not seen during training**.
- **Data split by compound** (stratified by chemical class), so the test set measures performance on new molecules and not on new temperatures of known ones.
- **Best model: tuned XGBoost**, ahead of the DNN on every test metric.
- **Part 1** uses CO2 data to show why the physics matters: modelling $\ln P$ against $1/T$ (Clausius-Clapeyron) turns a poor linear fit into an almost perfect one.

## Results (test set, 8,610 rows, unseen compounds)

| Model | MAE | RMSE | $R^2$ |
|---|---|---|---|
| **XGBoost (tuned)** | **1.230** | **1.968** | **0.781** |
| XGBoost (default) | 1.270 | 2.008 | 0.772 |
| Random Forest | 1.353 | 2.264 | 0.710 |
| DNN (hidden size 50) | 1.634 | 2.516 | 0.642 |
| Decision Tree | 1.650 | 2.707 | 0.586 |

MAE and RMSE are in $\ln P$ units: an error of $x$ means a factor of $e^{x}$ in the pressure, so the tuned XGBoost is typically off by a factor of about 3.4 and the DNN by about 5.1. MAPE is not used as a main metric, because $\ln P$ crosses zero (see `info.md`).

**Findings**
- The model family mattered much more than tuning: DNN to XGBoost gave about 0.40 lower MAE, and tuning XGBoost gave about 0.04.
- The model works well in the dense middle of the data and fails at very low pressures, for unusual classes (inorganics, amides, acids) and for some very small molecules whose fingerprints are empty.
- Details: [REPORT.md](REPORT.md) and [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

## Project structure

```
.
├── README.md
├── REPORT.md              full report: methods, results, conclusions
├── ERROR_ANALYSIS.md      where the models fail
├── info.md                theory notes: concepts, models, metrics, validation
├── LICENSE
├── requirements.txt
├── data/                  CO2_data.csv, Vapor_pressures.csv (not tracked, see below)
├── models/                final DNN and XGBoost, scaler, settings
├── notebook/
│   ├── eda.ipynb          Part 1 and data exploration
│   ├── part2.ipynb        fingerprints, split, DNN, early stopping, tuning
│   └── other_models.ipynb Decision Tree, Random Forest, XGBoost, test comparison
└── src/
    └── model.py           the DNN class
```

_Check the notebook descriptions against what each notebook actually contains._

## Data

Both files come from the book's repository: [`references/`](https://github.com/edgarsmdn/MLCE_book/tree/main/references).

| File | Content |
|---|---|
| `CO2_data.csv` | CO2 saturation data (T, P_vap, H_vap, densities) |
| `Vapor_pressures.csv` | vapor pressures of many compounds, with SMILES, chemical class and subclass |

The data folder is not tracked by git. Download both files into `data/` before running the notebooks.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Main packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `rdkit`, `datamol`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `tqdm`.

Run the notebooks from the `notebook/` folder, in the order: `eda`, `part2`, `other_models`.

## Method in brief

1. **Features:** the SMILES string is converted to a 2048-bit topological fingerprint, and temperature (K) is standardized with a scaler fitted on the training set only.
2. **Target:** $\ln P_{vap}$ (kPa), which reduces the skewness from 3.3 to -1.57 and follows the Clausius-Clapeyron form.
3. **Split:** by compound within each chemical class: 26,310 train, 6,810 validation and 8,610 test rows.
4. **DNN:** 2 hidden layers with ReLU, trained with Adam and early stopping, hidden size chosen from 3, 10, 20 and 50 on the validation set.
5. **Tree models:** compared with default settings on the validation set, then XGBoost tuned with `RandomizedSearchCV` and `GroupKFold` (grouped by compound).
6. **Evaluation:** all models evaluated once on the test set, after every decision was made.

## Saved models

| File | Content |
|---|---|
| `models/dnn_vle.pt` | DNN weights (hidden size 50) |
| `models/xgb_tuned.json` | tuned XGBoost |
| `models/temperature_scaler.joblib` | scaler for temperature |
| `models/config.json`, `models/xgb_config.json` | settings and tuned parameters |

Predictions need the **same fingerprint** (topological, 2048 bits) and the **saved scaler** applied to the temperature in K. The DNN class is defined in `src/model.py`.

## Limitations

- The best test $R^2$ is 0.78, which is useful for rough estimates and screening but not accurate enough for design calculations.
- Errors are large at very low pressures and temperatures, and for unusual chemistry. There is no warning yet for molecules far from the training data.
- Results come from a single split and a single random seed for each model.

## Possible extensions

- Similarity score (applicability domain) with a warning for new molecules
- Streamlit app: SMILES and temperature in, predicted vapor pressure out
- The same network in Keras/TensorFlow, for comparison

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

Tutorial and datasets from the [MLCE book](https://edgarsmdn.github.io/MLCE_book/) by Edgar Sanchez Medina, Antonio del Rio Chanona and Caroline Ganzer.

## Live demo

Try the app: **https://vle-vapor-pressure.streamlit.app**

Enter a molecule as a SMILES string and a temperature in K, and the app predicts its vapor pressure in kPa using the tuned XGBoost model (test MAE 1.23 on ln P). It also plots the vapor pressure curve over a temperature range you choose.

**Limitations**
- Very small molecules (for example ethanol or methane) give an empty fingerprint, so the model cannot tell them apart. The app shows a warning for these, and the number should not be trusted.
- Predictions are weakest for very low pressures and for classes that are rare in the training data, such as inorganics, amides and acids.
- The app takes SMILES strings only (compound names are planned).

**Run it locally**
```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```