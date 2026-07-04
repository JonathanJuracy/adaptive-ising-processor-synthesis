# Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Model

This repository contains the source code associated with the master's dissertation and the paper:

**A Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Model**

The script implements simulation routines based on the Ising model and generates the final plots from the consolidated results used in the work.

## Contents

* `ising_system.py`: main script containing the Ising model formulation, probabilistic methods, and plot generation.
* `requirements.txt`: Python libraries required to run the code.
* `.gitignore`: files and folders that do not need to be versioned.

When the script is executed, the `figures_out/` folder is automatically created with the plots and the `results.csv` table.

## Requirements

* Python 3.10 or higher
* NumPy
* Pandas
* Matplotlib

## How to Run

In the terminal, inside the project folder:

```bash
pip install -r requirements.txt
python ising_system.py
```

The output files will be generated in:

```text
figures_out/
```

## General Code Structure

The code is organized into four main blocks:

1. Configuration of the problems, algorithms, and consolidated results.
2. Theoretical core with functions for Ising energy, p-bit updates, and Gibbs, SA, SQA, and Cluster dynamics.
3. Organization of the experimental data into a table.
4. Generation of the final plots in PNG, PDF, and SVG formats.

## Author

Jonathan Juracy Carneiro da Silva
Graduate Program in Microelectronics — PGMICRO/UFRGS

## Note

This repository is intended for academic purposes and accompanies the results presented in the master's dissertation.
