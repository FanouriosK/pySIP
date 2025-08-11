from dataclasses import dataclass

from ..base import RCModel


@dataclass
class TiTe_RiaRieReaAw(RCModel):
    """
    Second order CTSM-R model with indoor (Ti) and envelope (Te) temperatures.

    This model is a simplified representation of a building, including resistances
    from indoor-to-ambient (Ria), indoor-to-envelope (Rie), and envelope-to-ambient (Rea).
    """

    # Define the state variables of the model
    states = [
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xe", "envelope temperature"),
    ]

    # Define the parameters to be identified
    params = [
        # Physical parameters
        ("THERMAL_RESISTANCE", "Ria", "between the indoor and the ambient"),
        ("THERMAL_RESISTANCE", "Rie", "between the indoor and the envelope"),
        ("THERMAL_RESISTANCE", "Rea", "between the envelope and the ambient"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor air"),
        ("THERMAL_CAPACITY", "Ce", "of the envelope"),
        ("SOLAR_APERTURE", "Aw", "of the windows (to indoor)"),
        ("SOLAR_APERTURE", "Ae", "of the envelope"),

        # Stochastic parameters (process noise)
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_e", "of the envelope dynamic"),

        # Stochastic parameters (measurement noise)
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurement"),

        # Initial state estimates
        ("INITIAL_MEAN", "x0_i", "of the indoor temperature"),
        ("INITIAL_MEAN", "x0_e", "of the envelope temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the indoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_e", "of the envelope temperature"),
    ]

    # Define the external inputs to the model
    inputs = [
        ("TEMPERATURE", "Ta", "ambient outdoor temperature"),
        ("POWER", "Phi_s", "solar irradiance"),
        ("POWER", "Phi_h", "HVAC system heat"),
    ]

    # Define the model outputs (what is measured)
    outputs = [("TEMPERATURE", "xi", "indoor temperature")]

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        super().__post_init__()

    def set_constant_continuous_ssm(self):
        """Set the time-invariant parts of the state-space model matrices."""
        # The output y is the first state, xi (indoor temperature)
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self):
        """
        Update the continuous state-space matrices A, B, Q, R, x0, P0
        based on the current parameter values.
        """
        # Unpack the parameters by their defined order
        (
            Ria, Rie, Rea, Ci, Ce, Aw, Ae,
            sigw_i, sigw_e, sigv,
            x0_i, x0_e, sigx0_i, sigx0_e,
            *_,
        ) = self.parameters.theta

        # --- System Matrix A ---
        # Derived from the differential equations for dTi/dt and dTe/dt
        # dTi/dt = (1/Ci)*( (Te-Ti)/Rie + (Ta-Ti)/Ria + Aw*Phi_s + Phi_h )
        # dTe/dt = (1/Ce)*( (Ti-Te)/Rie + (Ta-Te)/Rea + Ae*Phi_s )
        self.A[:] = [
            [-(1 / (Ci * Rie) + 1 / (Ci * Ria)), 1 / (Ci * Rie)],
            [1 / (Ce * Rie), -(1 / (Ce * Rie) + 1 / (Ce * Rea))],
        ]

        # --- Input Matrix B ---
        # Maps inputs [Ta, Phi_s, Phi_h] to the states
        self.B[:] = [
            # Effect on Ti
            [1 / (Ci * Ria), Aw / Ci, 1 / Ci],
            # Effect on Te
            [1 / (Ce * Rea), Ae / Ce, 0.0],
        ]

        # --- Process Noise Covariance Matrix Q ---
        # Represents the uncertainty in the state equations
        self.Q[self._diag] = [sigw_i, sigw_e]

        # --- Measurement Noise Covariance Matrix R ---
        # Represents the uncertainty in the measurements
        self.R[0, 0] = sigv

        # --- Initial State Mean x0 ---
        self.x0[:, 0] = [x0_i, x0_e]

        # --- Initial State Covariance P0 ---
        self.P0[self._diag] = [sigx0_i, sigx0_e]
