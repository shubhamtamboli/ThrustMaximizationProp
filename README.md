# ThrustMaximizationProp

**ThrustMaximizationProp** is an automated pipeline for optimizing propeller geometry to maximize thrust under specific cruise conditions. It leverages **Differential Evolution (DE)** algorithms and **Bezier curve parameterization** to generate smooth, efficient blade shapes, validated using the **XROTOR** aerodynamic solver.

## 🚀 Features

- **Advanced Optimization**: Uses the `pymoo` library's Differential Evolution (DE) algorithm to explore the design space and avoid local optima.
- **Smooth Geometry**: Parameterizes chord and twist distributions using Bezier curves (via Control Points) to ensure manufacturable and aerodynamically smooth blade shapes.
- **Aerodynamic Analysis**: Integrates directly with **XROTOR** to evaluate propeller performance (Thrust, Power, Efficiency).
- **Multi-Start Strategy**: Supports multiple optimization runs with random seeds to ensure robustness and find the global optimum.
- **Configurable**: Fully driven by a JSON configuration file for easy adjustment of operating conditions, geometry bounds, and optimization settings.

## 🛠️ Prerequisites

- **Python 3.8+**
- **XROTOR**: The `xrotor` executable must be installed and accessible.
- Python dependencies:
  ```bash
  pip install numpy scipy pymoo
  ```

## 📂 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/shubhamtamboli/ThrustMaximizationProp.git
    cd ThrustMaximizationProp
    ```
2.  Ensure `xrotor` is correctly set up in your environment or the path is configured in the interface scripts.

## 🏃 Usage

Run the optimization pipeline using `main.py` and provide a configuration file:

```bash
python main.py --file thrust-cruise.json --generations 100
```

### Arguments

- `-f`, `--file`: (Required) Filename of the JSON configuration located in `processing/inputs/`.
- `--generations`: (Optional) Number of generations for the optimizer (default: 10000).

## ⚙️ Configuration

The optimization is controlled by a JSON file (e.g., `processing/inputs/thrust-cruise.json`).

### Structure

```json
{
    "optimization": {
        "n_gen": 100,          // Number of generations
        "pop_size": 50,        // Population size
        "n_restarts": 5        // Number of independent runs
    },
    "operatingConditions": {
        "V": 20.0,             // Cruise velocity (m/s)
        "rho": 1.225,          // Air density (kg/m^3)
        "rpm": 12000,          // Propeller RPM
        "maxPower": 50000.0,   // Max power constraint (W)
        "requiredThrust": 0.0  // Min thrust constraint (N)
    },
    "geometryGlobal": {
        "diameter": 0.30,      // Propeller diameter (m)
        "bladeCount": 8,       // Number of blades
        "rStations": [...]     // Radial stations for evaluation
    },
    "geometryBounds": {
        "chordMin": 0.05,      // Min chord (fraction of Diameter)
        "chordMax": 0.15,      // Max chord (fraction of Diameter)
        "pitchDegMin": 0.0,    // Min pitch (degrees)
        "pitchDegMax": 40.0    // Max pitch (degrees)
    }
}
```

## 📊 Output

- **Console**: Real-time progress of the optimization (Generations, Best Thrust, Constraint Violations).
- **Results**: The best design is saved to `processing/outputs/` (or configured output path), typically including:
    - Optimized Control Points
    - Performance Metrics (Thrust, Efficiency)
    - Geometry Data (Chord/Pitch distributions)

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

[MIT License](LICENSE)
