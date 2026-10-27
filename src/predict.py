import datamol as dm
# Function to turn smiles to mol
def smiles_to_mol(smiles):
    """Convert a SMILES string into an RDKit molecule.

    Args:
        smiles: SMILES string of the molecule, for example "CCO".

    Returns:
        The RDKit molecule object.

    Raises:
        ValueError: If the SMILES string cannot be parsed.
    """
    mol = dm.to_mol(smiles)
    if mol is None:
        raise ValueError("Please enter a valid SMILES string.")
    return mol

def mol_to_fingerprint(mol, fp_type='topological'):
    """Turns molecule into fingerprints.
    Args:
        Molecules, the return from the above function.
        fp_type: Fingerprint type used by datamol. The default,
            'topological', is the hashed topological torsion fingerprint:
            it encodes paths of four consecutive bonded heavy atoms and
            hashes them into 2048 bits (1 = pattern present). Molecules
            with fewer than four heavy atoms in a path, such as ethanol
            or methane, get an all-zero fingerprint.

    Returns:
        NumPy array of 2048 values (0 or 1).
    Returns:
        fingerprint: NumPy array of 0s and 1s (2048 bits).    
    """
    return dm.to_fp(mol, fp_type=fp_type)    


def too_few_bits(fingerprint, min_bits=1):
    """Check whether a fingerprint has too few set bits to be reliable.

    Very small molecules (for example ethanol or methane) get an empty
    topological fingerprint, so the model cannot tell them apart and its
    prediction is not trustworthy.

    Args:
        fingerprint: NumPy array of 0s and 1s (2048 bits).
        min_bits: Minimum number of set bits required. Default is 1,
            which only flags an empty fingerprint.

    Returns:
        True if the fingerprint has fewer than `min_bits` bits set,
        otherwise False.
    """
    n_bits = fingerprint.sum()
    return n_bits < min_bits


import numpy as np

def build_input(fingerprint, temperature_K, scaler):
    """Build the one-row model input: fingerprint + scaled temperature.

    Args:
        fingerprint: array of 2048 values from `mol_to_fingerprint`.
        temperature_K: temperature in Kelvin.
        scaler: the fitted StandardScaler saved after training.

    Returns:
        Array of shape (1, 2049).
    """
    T_scaled = scaler.transform([[temperature_K]])            # shape (1, 1)
    X = np.hstack((fingerprint.reshape(1, -1), T_scaled))     # shape (1, 2049)
    return X

from xgboost import XGBRegressor
import numpy as np
model = XGBRegressor()
# model.load_model("../models/models/xgb_tuned.json")

def predict_pressure(X, model):
    """Predicts the pressure and give two output in lnP and P

    Args:
        X : array of 2049 values from `mol_to_fingerprint` and temperature.
        model: the best model that was saved which is XGBoost in this case

    Returns:
        ln_p : tha raw value of the prediction
        the pressure value after the conversion, kPa    
    """
    ln_p = model.predict(X)[0]
    return ln_p, float(np.exp(ln_p))

def predict_from_smiles(smiles, temperature_K, model, scaler, min_bits=1):
    """Predict the vapor pressure of a molecule at a given temperature.

    Args:
        smiles: SMILES string of the molecule.
        temperature_K: temperature in Kelvin.
        model: the fitted XGBoost model.
        scaler: the fitted temperature scaler.
        min_bits: minimum set bits for the fingerprint to be considered reliable.

    Returns:
        Dictionary with the pressure in kPa, ln P, the number of set
        fingerprint bits, and a flag that is True when the prediction
        should not be trusted (too few bits).

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    mol = smiles_to_mol(smiles)
    fingerprint = mol_to_fingerprint(mol)
    X = build_input(fingerprint, temperature_K, scaler)
    ln_p, p = predict_pressure(X, model)
    return {
        "pressure_kPa": float(p),
        "ln_p": float(ln_p),
        "n_bits": int(fingerprint.sum()),
        "warn": bool(too_few_bits(fingerprint, min_bits)),
    }