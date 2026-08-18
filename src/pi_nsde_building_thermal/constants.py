"""Physical constants and default building coefficients.

Thermal dynamics are integrated in hours with:
    C [kWh/K] * dT/dt [K/h] = Q [kW]
"""

# Dry air
RHO_AIR_KG_M3 = 1.2
CP_AIR_J_KG_K = 1006.0
LATENT_HEAT_VAPOR_J_KG = 2.45e6
ATMOS_PRESSURE_PA = 101_325.0

# Magnus-Tetens saturation vapor pressure
MAGNUS_A = 610.94
MAGNUS_B = 17.625
MAGNUS_C = 243.04
WATER_AIR_MASS_RATIO = 0.62198

# Known wind-driven infiltration multiplier on UA: UA_eff = UA * (1 + k * v)
# k is not learned; it lets wind enter the energy balance without aliasing R.
WIND_INFILTRATION_PER_MPS = 0.04

# Filter Q_int(t0). Match the plant occupancy initial condition so the
# identifier does not start with a 0.1 kW occupancy offset the plant never had.
FILTER_Q0_KW = 0.6
