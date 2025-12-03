import numpy as np
import matplotlib.pyplot as plt
import os
from .geometry_param import decode_design
from .xrotor_interface import XRotorWrapper


def save_results(res, cfg, outdir="processing/outputs"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Best design vector
    x_best = res.X

    if x_best is None:
        print("No feasible solution found (res.X is None).")
        return

    chords, betas = decode_design(
        x_best, cfg["geometryBounds"], cfg["geometryGlobal"]
    )

    # Calculate efficiency
    xr = XRotorWrapper(cfg)
    perf = xr.evaluate(chords, betas)
    eta = perf.get("eta", 0.0)

    # Save geometry
    np.savetxt(
        os.path.join(outdir, "best_chord_pitch.csv"),
        np.vstack((chords, betas)).T,
        delimiter=",",
        header=f"Global_Efficiency:{eta:.4f}\nChord(m),PitchLE(deg)",
        comments=""
    )

    # Simple convergence plot if algorithm history is available
    if hasattr(res.algorithm, "opt"):
        try:
            hist = res.algorithm.opt.get("F")
            hist = np.array(hist).flatten()
            thrust_hist = -hist

            plt.figure()
            plt.plot(thrust_hist)
            plt.xlabel("Iteration")
            plt.ylabel("Thrust (arb. units)")
            plt.title("Thrust Convergence")
            plt.grid(True)
            plt.savefig(os.path.join(outdir, "thrust_convergence.png"))
            plt.close()
        except Exception:
            pass

    print("\nOptimization Finished!")
    print("Optimal objective (max thrust) =", -res.F[0])
    print("Results saved to:", outdir)
