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