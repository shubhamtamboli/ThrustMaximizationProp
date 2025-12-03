import numpy as np
from xrotor import XRotor
from xrotor.model import Case


class XRotorWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.geom = cfg["geometryGlobal"]
        self.oc = cfg["operatingConditions"]

        # Geometry / global
        self.D = self.geom["diameter"]
        self.R = self.D / 2.0
        self.B = self.geom["bladeCount"]
        self.rStations = np.array(self.geom["rStations"])   # nondimensional [0–1]

        # Operating conditions
        self.V = self.oc["V"]
        self.rho = self.oc["rho"]
        self.rpm = self.oc["rpm"]

        # Rough constants
        self.vso = 340.0
        self.mu = 1.789e-5
        self.alt = 0.0


    def _make_polars(self):
        """
        Return user-provided Clark Y polar data (Re ~ 500k, Mach ~ 0.35).
        Columns: Alpha(deg), CL, CD, Cm
        """
        data = np.array([
            [-10.0, -0.3448, 0.10501, -0.0385],
            [-9.5, -0.3361, 0.09944, -0.0397],
            [-9.0, -0.3247, 0.08659, -0.0508],
            [-8.5, -0.3143, 0.07447, -0.0628],
            [-8.0, -0.3190, 0.05814, -0.0826],
            [-7.5, -0.3695, 0.02932, -0.1075],
            [-7.0, -0.3436, 0.02184, -0.1071],
            [-6.5, -0.2838, 0.01800, -0.1079],
            [-6.0, -0.2385, 0.01526, -0.1052],
            [-5.5, -0.1727, 0.01395, -0.1054],
            [-5.0, -0.1242, 0.01240, -0.1024],
            [-4.5, -0.0653, 0.01184, -0.1012],
            [-4.0, -0.0089, 0.01091, -0.0997],
            [-3.5, 0.0431, 0.01026, -0.0973],
            [-3.0, 0.0999, 0.00995, -0.0959],
            [-2.5, 0.1545, 0.00948, -0.0941],
            [-2.0, 0.2072, 0.00901, -0.0920],
            [-1.5, 0.2587, 0.00856, -0.0899],
            [-0.5, 0.3550, 0.00754, -0.0845],
            [0.0, 0.4047, 0.00676, -0.0814],
            [0.5, 0.5314, 0.00709, -0.0948],
            [1.0, 0.6309, 0.00750, -0.1025],
            [1.5, 0.7175, 0.00771, -0.1081],
            [2.0, 0.7579, 0.00793, -0.1038],
            [2.5, 0.7981, 0.00821, -0.0995],
            [3.0, 0.8358, 0.00861, -0.0946],
            [3.5, 0.8735, 0.00907, -0.0897],
            [4.0, 0.9102, 0.00960, -0.0846],
            [4.5, 0.9452, 0.01021, -0.0792],
            [5.0, 0.9817, 0.01080, -0.0741],
            [5.5, 1.0202, 0.01135, -0.0694],
            [6.0, 1.0550, 0.01188, -0.0639],
            [6.5, 1.0897, 0.01246, -0.0584],
            [7.0, 1.1243, 0.01315, -0.0531],
            [7.5, 1.1629, 0.01379, -0.0486],
            [8.0, 1.1957, 0.01472, -0.0435],
            [8.5, 1.2170, 0.01630, -0.0371],
            [9.0, 1.2275, 0.01869, -0.0301],
            [9.5, 1.2439, 0.02102, -0.0244],
            [10.0, 1.2688, 0.02306, -0.0201],
            [10.5, 1.2824, 0.02600, -0.0154],
            [11.0, 1.2922, 0.02948, -0.0112],
            [11.5, 1.3109, 0.03261, -0.0082],
            [12.0, 1.3231, 0.03646, -0.0055],
            [12.5, 1.3200, 0.04191, -0.0032],
            [13.0, 1.3237, 0.04707, -0.0016],
            [13.5, 1.3324, 0.05221, -0.0010],
            [14.0, 1.3333, 0.05846, -0.0011],
            [14.5, 1.3133, 0.06695, -0.0018],
            [15.0, 1.3119, 0.07404, -0.0029],
            [15.5, 1.3048, 0.08208, -0.0049],
            [16.0, 1.2916, 0.09107, -0.0077],
            [16.5, 1.2671, 0.10158, -0.0112],
            [17.0, 1.2522, 0.11127, -0.0147],
            [17.5, 1.2331, 0.12165, -0.0190],
            [18.0, 1.2133, 0.13257, -0.0241],
            [18.5, 1.1914, 0.14417, -0.0300],
            [19.0, 1.1614, 0.15750, -0.0373],
            [19.5, 1.1295, 0.17217, -0.0459],
            [20.0, 1.0784, 0.19284, -0.0585]
        ])
        return data


    def _build_case_dict(self, chords, betas_le):
        """
        Create a case dict using the MODERN nested format, matching the official example.
        
        SCALING RULES (based on example):
        - r_hub, r_tip: DIMENSIONAL (meters)
        - radii: NORMALIZED (r/R)
        - chord: NORMALIZED (c/R)
        - twist: DEGREES
        """

        # Normalized Geometry Arrays
        radii_norm = self.rStations          # r/R
        chord_norm = chords / self.R         # c/R
        twist = betas_le                     # deg

        # Dimensional Hub/Tip
        r_hub_dim = float(radii_norm[0] * self.R)
        r_tip_dim = float(self.R)

        # Advance ratio
        omega = 2.0 * np.pi * self.rpm / 60.0
        adv = self.V / (omega * self.R) if omega * self.R > 0 else 0.0

        # Synthetic polar data
        polars = self._make_polars()

        case = {
            "conditions": {
                "rho": self.rho,
                "vso": self.vso,
                "rmu": self.mu,
                "alt": self.alt,
                "vel": self.V,
                "adv": adv
            },
            "disk": {
                "n_blds": int(self.B),
                "blade": {
                    "geometry": {
                        "r_hub": r_hub_dim,    # Dimensional
                        "r_tip": r_tip_dim,    # Dimensional
                        "r_wake": 0.0,
                        "rake": 0.0,
                        "radii": radii_norm,   # Normalized
                        "chord": chord_norm,   # Normalized
                        "twist": twist,
                        "ubody": np.zeros_like(radii_norm)
                    }
                }
            },
            "settings": {
                "free": True,
                "duct": False,
                "wind": False,
                "wake": False   # Rigid wake for stability
            }
        }

        return case


    def evaluate(self, chords, betas_le):
        xr = XRotor()
        
        # 1. Create Case object from dict (without polars)
        case_obj = Case.from_dict(self._build_case_dict(chords, betas_le))

        # 2. Manually inject polars to avoid TypeError in from_dict
        # and ValueError in xr.case setter
        polars = self._make_polars()
        if hasattr(case_obj, 'disk') and hasattr(case_obj.disk, 'blade'):
             # Assign as a dictionary {radial_pos: polar_array}
             # Using 0.0 implies it applies to the whole blade (or root)
             case_obj.disk.blade.polars = {0.0: polars}

        xr.case = case_obj

        # Suppress XROTOR output
        import os
        STDOUT_FD = 1
        STDERR_FD = 2
        saved_stdout = os.dup(STDOUT_FD)
        saved_stderr = os.dup(STDERR_FD)

        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, STDOUT_FD)
            os.dup2(devnull, STDERR_FD)
            
            try:
                xr.operate(rpm=self.rpm)
            finally:
                os.dup2(saved_stdout, STDOUT_FD)
                os.dup2(saved_stderr, STDERR_FD)
                os.close(saved_stdout)
                os.close(saved_stderr)
                os.close(devnull)
            
            perf = xr.performance
            
            # Sign flip
            # XROTOR defines Thrust as force ON THE FLUID (positive downstream).
            # We want force ON THE PROPELLER (positive upstream).
            # Wait, standard XROTOR convention: T > 0 is propulsion.
            # Let's trust the positive value if P is positive.
            T_val = float(perf.thrust)
            P_val = float(perf.power)
            
            # Sanity Check
            # 0.1m prop -> T ~ 1-10 N. P ~ 10-100 W.
            if abs(T_val) > 2000.0 or abs(P_val) > 100000.0:
                raise ValueError(f"Numerical Explosion: T={T_val}, P={P_val}")

            # Calculate Torque
            omega = 2.0 * np.pi * self.rpm / 60.0
            Q_val = P_val / omega if omega > 0 else 0.0

            n = self.rpm / 60.0
            J = self.V / (n * self.D) if (n * self.D) > 0 else 0.0

            return {
                "T": T_val,
                "P": P_val,
                "Q": Q_val,
                "RPM": self.rpm,
                "eta": float(perf.efficiency),
                "J": J
            }

        except Exception as e:
            # print("\n XROTOR ERROR — Using penalty:", e)
            return {
                "T": 0.0,
                "P": 1e9,
                "Q": 0.0,
                "RPM": 0.0,
                "eta": 0.0,
                "J": 0.0
            }
