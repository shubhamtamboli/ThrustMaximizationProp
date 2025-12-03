import numpy as np
from .geometry_param import decode_design
from .xrotor_interface import XRotorWrapper

M_TIP_LIMIT = 0.75  # maximum allowable tip Mach number


def compute_tip_mach(D, rpm, V, a=340.0):
    R = D / 2.0
    omega = 2.0 * np.pi * rpm / 60.0
    V_tip = np.sqrt((omega * R)**2 + V**2)
    return V_tip / a


class ThrustCruiseObjective:
    def __init__(self, cfg):
        self.cfg = cfg
        self.Pmax = cfg["operatingConditions"]["maxPower"]
        self.xr = XRotorWrapper(cfg)

        # Required thrust constraint
        self.T_req = cfg["operatingConditions"].get("requiredThrust", 5.0)
        
        # Minimum efficiency constraint
        self.min_eta = cfg["operatingConditions"].get("minEfficiency", 0.5)

    def __call__(self, x):
        # Decode geometry: fixed chord, optimized pitch
        chords, betas_le = decode_design(
            x, self.cfg["geometryBounds"], self.cfg["geometryGlobal"]
        )

        # 1) Run XROTOR
        try:
            perf = self.xr.evaluate(chords, betas_le)
            T = perf["T"]
            P = perf["P"]
            
            # Debug print removed for cleaner output
            # import sys
            # print(f"  -> T={T:.2f} N, P={P:.2f} W", file=sys.stderr)

            if np.isnan(T) or np.isnan(P):
                raise ValueError("NaN returned from XROTOR")

        except:
            # Hard failure: treat as poor design
            T = 0.0
            P = 10.0 * self.Pmax

        # 2) Constraint calculations
        D = self.cfg["geometryGlobal"]["diameter"]
        V = self.cfg["operatingConditions"]["V"]
        rpm = self.cfg["operatingConditions"]["rpm"]

        M_tip = compute_tip_mach(D, rpm, V)

        # Constraint vector: g <= 0 is feasible
        g = np.zeros(6)

        # g[0] Power constraint (allow slight violation)
        g[0] = max(0, (P - self.Pmax) / self.Pmax)

        # g[1] Tip Mach constraint (softer)
        g[1] = max(0, (M_tip - M_TIP_LIMIT) / M_TIP_LIMIT)

        # g[2] Required thrust (very soft early)
        # Normalize by abs(T_req) to handle negative requirements correctly
        g[2] = max(0, (self.T_req - T) / (abs(self.T_req) + 1e-6))

        # g[3] Minimum efficiency
        # eta = T * V / P (if P > 0)
        if P > 1e-6:
            eta = T * V / P
        else:
            eta = 0.0
        
        # We want eta >= min_eta  =>  min_eta - eta <= 0
        g[3] = max(0, self.min_eta - eta)

        # --- Geometric Constraints ---
        
        # g[4] Taper Constraint: Tip Chord <= 0.7 * Root Chord
        # c_tip <= 0.7 * c_root  =>  c_tip - 0.7*c_root <= 0
        c_root = chords[0]
        c_tip = chords[-1]
        g[4] = max(0, (c_tip - 0.7 * c_root) / c_root)

        # g[5] Washout Constraint: Tip Pitch <= Root Pitch
        # beta_tip <= beta_root  =>  beta_tip - beta_root <= 0
        beta_root = betas_le[0]
        beta_tip = betas_le[-1]
        g[5] = max(0, (beta_tip - beta_root) / 10.0)  # Normalize by 10 deg

        # Objective: maximize thrust → minimize -T
        f = -T

        return f, g

    def get_performance(self, x):
        """
        Helper to retrieve full performance dict for a given design vector x.
        Used by the Display class to show efficiency.
        """
        chords, betas_le = decode_design(
            x, self.cfg["geometryBounds"], self.cfg["geometryGlobal"]
        )
        return self.xr.evaluate(chords, betas_le)
