import json
import argparse
from src.optimizer import run_optimization
from src.output_process import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Input JSON filename")
    parser.add_argument("--generations", type=int, default=10000, help="Number of generations")
    args = parser.parse_args()

    input_path = f"processing/inputs/{args.file}"
    cfg = json.load(open(input_path))

    # Read optimization settings
    opt_cfg = cfg.get("optimization", {})
    n_gen = opt_cfg.get("n_gen", 100)
    pop_size = opt_cfg.get("pop_size", 50)
    n_restarts = opt_cfg.get("n_restarts", 5)  # Default to 5 restarts

    print(f"Running optimization using: {args.file}")
    print(f"Generations: {n_gen}, Population: {pop_size}, Restarts: {n_restarts}")
    
    best_res = None
    best_f = float("inf")
    
    results_summary = []

    for i in range(n_restarts):
        print(f"\n--- Run {i+1}/{n_restarts} ---")
        # Use random seed for each run
        res = run_optimization(cfg, n_gen=n_gen, pop_size=pop_size, seed=None)
        
        # Store result
        if res.X is not None:
            thrust = -res.F[0]
            results_summary.append((i+1, thrust))
            print(f"  -> Run {i+1} Result: T = {thrust:.2f} N")
            
            # Update best
            if res.F[0] < best_f:
                best_f = res.F[0]
                best_res = res
                print(f"  -> New Best Found!")
        else:
            results_summary.append((i+1, 0.0))
            print(f"  -> Run {i+1} Failed (No feasible solution)")

    print("\n" + "="*40)
    print("MULTI-START SUMMARY")
    print("="*40)
    print(f"{'Run':<5} | {'Thrust (N)':<15}")
    print("-" * 25)
    for run_id, t_val in results_summary:
        print(f"{run_id:<5} | {t_val:.2f}")
    print("="*40)

    if best_res is not None:
        print(f"\nBest Result: T = {-best_f:.2f} N")
        save_results(best_res, cfg)
    else:
        print("\nNo feasible solution found in any run.")


if __name__ == "__main__":
    main()
