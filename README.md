Claro. Use esta versão mais resumida para copiar e colar no:

```text
adaptive_ising_tool/README.md
```

````markdown
# Adaptive Ising Tool

This folder contains the adaptive Ising simulation tool developed for the master's dissertation:

**Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Model**

The tool allows the user to define an optimization problem, preprocess the input data, build the corresponding Ising Hamiltonian, configure the probabilistic machine, select an update algorithm, and run the simulation.

## Contents

* `ising_adaptive_tool.py`: main script of the adaptive Ising tool.
* `examples/example_maxcut_problem.json`: example input problem.
* `README.md`: basic documentation for this tool.

## Requirements

* Python 3.10 or higher
* NumPy
* Pandas
* Matplotlib

## How to Run

Inside the `adaptive_ising_tool/` folder, run:

```bash
python ising_adaptive_tool.py --input examples/example_maxcut_problem.json --output results --algorithm auto --iterations 500
````

To run all available algorithms:

```bash
python ising_adaptive_tool.py --input examples/example_maxcut_problem.json --output results_all --algorithm all --iterations 500
```

## Supported Problems

The current version supports:

* `custom_ising`
* `maxcut`
* `coloring`
* `tsp`
* `max2sat`
* `matching`
* `segmentation`

## Output Files

The simulation generates:

* `ising_J.csv`
* `ising_h.csv`
* `simulation_results.csv`
* `energy_trace.csv`
* `energy_trace.png`
* `summary.json`

## General Workflow

The tool follows the dissertation flow:

1. input data and problem definition;
2. preprocessing;
3. Ising Hamiltonian construction;
4. probabilistic machine configuration;
5. algorithm selection;
6. simulation;
7. result generation.

## Implemented Algorithms

* Gibbs Sampling
* Simulated Annealing
* Simulated Quantum Annealing
* Cluster-based updates

## Author

Jonathan Juracy Carneiro da Silva
Graduate Program in Microelectronics — PGMICRO/UFRGS

## Note

This tool is intended for academic purposes and accompanies the methodology presented in the master's dissertation.

````

Mensagem de commit:

```text
Add dissertation tool README
````
