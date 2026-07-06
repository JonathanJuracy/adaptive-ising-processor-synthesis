# Adaptive Ising Processor Synthesis

This repository contains the source code used to support the development and validation of an adaptive probabilistic processor synthesis flow based on the Ising model.

The project is organized into two main folders:

## Repository structure

```text
adaptive-ising-processor-synthesis/
├── adaptive_ising_tool/
│   └── Source code of the adaptive Ising-based synthesis tool.
│
├── article_simulation/
│   └── Scripts and data used to generate the simulation results and plots presented in the article.
│
└── README.md
```

## adaptive_ising_tool

This folder contains the main tool implementation. It includes the routines responsible for mapping combinatorial optimization problems to the Ising model, defining the required number of probabilistic elements, and selecting the update strategy used during simulation.

The tool is intended to support experiments with probabilistic computing architectures based on p-bits, MTJs, and Ising-machine-inspired optimization flows.

## article_simulation

This folder contains the scripts used to reproduce the experimental results shown in the article. It includes the simulation setup, benchmark configuration, consolidated numerical results, and plot generation routines.

The results cover representative optimization problems such as TSP, graph coloring, SAT, matching, segmentation, and Max-Cut.

## Requirements

Each folder may include its own `requirements.txt` file with the Python packages required to execute the corresponding scripts.

A typical setup is:

```bash
pip install -r requirements.txt
```

or, when inside a specific folder:

```bash
cd article_simulation
pip install -r requirements.txt
```

## General purpose

The repository provides a compact and reproducible implementation of an adaptive Ising-based simulation flow, supporting both tool development and article result generation.

## Citation

If this repository is used as a reference, please cite the related work:

```bibtex
@article{da2026tool,
  title={A Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Model},
  author={da Silva, Jonathan Juracy Carneiro and Gobatto, Leonardo R. and Azambuja, Jose Rodrigo},
  journal={arXiv preprint arXiv:2606.19533},
  year={2026}
}
```
