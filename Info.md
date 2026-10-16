# Info

## Critical point

The critical point is the end of the vapor-liquid coexistence curve, defined by the critical temperature $T_c$ and critical pressure $P_c$.

- The liquid and vapor densities become equal, so the two phases are indistinguishable.
- The enthalpy of vaporization goes to zero: $\Delta H_{vap} \to 0$ as $T \to T_c$.
- Above $T_c$ there is a single supercritical fluid; no pressure can condense it into a separate liquid.

**In the CO2 data:** the row where `H_vap = 0` is the critical point (about 31.1 °C and 7391 kPa).

## Clapeyron and Clausius-Clapeyron equations

Along the vapor-liquid coexistence curve, the exact Clapeyron equation relates the slope of the curve to the enthalpy and volume change of vaporization:

$$
\frac{dP}{dT} = \frac{\Delta H_{vap}}{T \, \Delta V}
$$

Clausius-Clapeyron is the approximation obtained by assuming:

1. the liquid volume is negligible, so $\Delta V \approx V_{vap}$
2. the vapor is an ideal gas, so $V_{vap} = RT/P$
3. $\Delta H_{vap}$ is constant

This gives

$$
\ln P = -\frac{\Delta H_{vap}}{R T} + c
$$

so $\ln P$ against $1/T$ is a straight line with slope $-\Delta H_{vap}/R$.

**Limitation:** all three assumptions fail near the critical point, where $\Delta H_{vap} \to 0$.

# Second part Neural Network

## Molecular fingerprint

A molecular fingerprint is a vector of 0s and 1s that describes a molecule's structure. It is computed from the SMILES string, and each bit marks whether a particular substructure (chain, ring, functional group, and so on) is present (1) or absent (0). In this project each fingerprint has 2048 bits.

- **Depends only on structure, not on temperature.** A compound has the same fingerprint at every temperature. Only the temperature and vapor pressure change between its rows in the dataset.
- **Model input.** The input to the neural network is the fingerprint (2048 bits) plus the scaled temperature as one extra column, giving 2049 features.

### Why the fingerprint is needed

Each compound has its own vapor pressure curve $P_{vap}(T)$. Temperature says *where on the curve* a point is, and the fingerprint says *which curve*. With temperature alone, methane, ethanol and decane would all look identical to the model.

A name or ID would say which compound it is, but not what it is like. The fingerprint describes the structure, so the network can learn patterns such as:

- longer carbon chains and larger molecules tend to have lower vapor pressure (stronger dispersion forces)
- hydrogen-bonding groups such as $-\text{OH}$ lower the vapor pressure
- structurally related molecules have similar fingerprints and similar behavior

Because a new molecule is described by the same kinds of substructures, the trained network can estimate its vapor pressure without any measurements on it.

### Comparison with the Antoine equation

The Antoine equation has constants $A$, $B$, $C$ fitted separately for each compound, so it cannot be used on a compound without data. The network uses one set of weights for all compounds and gets compound-specific behavior from the fingerprint.

### Consequences

- **Split by compound.** All rows of one compound share a fingerprint, so a random row split would put the same molecule in both train and test. The tutorial splits by compound so the test set measures performance on unseen molecules.
- **Similarity.** The similarity (Tanimoto) between two molecules is one number per pair, because it doesn't depend on temperature. It can be used to tell how close a new molecule is to the training compounds.

**Limitation:** the fingerprint only records presence or absence of substructures. _Add your own notes on what information about a molecule this loses._