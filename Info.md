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