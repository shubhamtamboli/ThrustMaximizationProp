import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize

from .objective_function import ThrustCruiseObjective


class ThrustCruiseProblem(Problem):
    def __init__(self, cfg):
        self.cfg = cfg
        # Use 5 control points for Chord and 5 for Twist
        self.n_cp = 5
        n_vars = 2 * self.n_cp

        super().__init__(
            n_var=n_vars,      # 5 chord CPs + 5 twist CPs
            n_obj=1,
            n_constr=6,
            xl=np.zeros(n_vars),
            xu=np.ones(n_vars),
        )

        self.obj = ThrustCruiseObjective(cfg)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        G = []
        for x in X:
            f, g = self.obj(x)
            F.append(f)
            G.append(g)

        out["F"] = np.array(F)
        out["G"] = np.array(G)


def run_optimization(cfg, n_gen=60, pop_size=16, seed=None):
    problem = ThrustCruiseProblem(cfg)
    # Use Differential Evolution (DE) for better global search robustness
    # DE is often better at finding the feasible region than CMA-ES for this type of problem.
    from pymoo.algorithms.soo.nonconvex.de import DE

    algo = DE(
        pop_size=pop_size,
        variant="DE/rand/1/bin",
        CR=0.9,
        F=0.8,
        dither="vector",
        jitter=False
    )

    print("\n" + "="*60)
    print("OPTIMIZATION PROGRESS LEGEND")
    print("="*60)
    print("n_gen    : Generation Number")
    print("n_eval   : Total Evaluations")
    print("cv (min) : Best Constraint Violation (0.0 = Valid Design)")
    print("f (min)  : Best Objective (-Thrust). E.g. -50.0 means 50N Thrust")
    print("="*60 + "\n")

    res = minimize(
        problem,
        algo,
        ("n_gen", n_gen),
        verbose=True,
        seed=seed
    )

    return res
