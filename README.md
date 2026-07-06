# Adaptive Ising Processor Synthesis

This repository contains the source code used to support the development and validation of an adaptive probabilistic processor synthesis flow based on the Ising model.

The project is organized into three main folders:

## Repository Structure

```text
adaptive-ising-processor-synthesis/
├── adaptive_ising_tool/
│   └── Source code of the adaptive Ising-based synthesis tool.
│
├── article_simulation/
│   └── Scripts and data used to generate the simulation results and plots presented in the article.
│
├── tangnano1k_fpga/
│   └── FPGA proof of concept implemented on the Tang Nano 1K board.
│
└── README.md
```

## adaptive_ising_tool

This folder contains the main tool implementation. It includes the routines responsible for mapping combinatorial optimization problems to the Ising model, defining the required number of probabilistic elements, and selecting the update strategy used during simulation.

The tool is intended to support experiments with probabilistic computing architectures based on p-bits, MTJs, and Ising-machine-inspired optimization flows.

## article_simulation

This folder contains the scripts used to reproduce the experimental results shown in the article. It includes the simulation setup, benchmark configuration, consolidated numerical results, and plot generation routines.

The results cover representative optimization problems such as TSP, graph coloring, SAT, matching, segmentation, and Max-Cut.

## tangnano1k_fpga

This folder contains the FPGA proof of concept developed using the Tang Nano 1K board. The implementation emulates four digital probabilistic bits, or p-bits, using a pseudo-random LFSR-based noise source and a simplified update rule inspired by the Ising model and the Max-Cut optimization problem.

The FPGA design exposes internal signals to external pins so they can be observed using a logic analyzer. These signals include the slow reference clock, update pulse, pseudo-random bit, four digital spin outputs, and a signal indicating when a favorable Max-Cut configuration is reached.

This implementation is not intended to physically reproduce the analog behavior of stochastic magnetic tunnel junctions. Instead, it provides a digital hardware demonstration of the basic concepts used in probabilistic Ising-inspired processors.

The Tang Nano 1K documentation is available at:

```text
https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-1K/Nano-1k.html
```

## Requirements

Each folder may include its own `requirements.txt` file or specific instructions required to execute the corresponding scripts.

For the Python-based simulations, a typical setup is:

```bash
pip install -r requirements.txt
```

or, when inside a specific folder:

```bash
cd article_simulation
pip install -r requirements.txt
```

For the FPGA proof of concept, the project can be synthesized and programmed using Gowin EDA with the Tang Nano 1K board configuration.

## General Purpose

The repository provides a compact and reproducible implementation of an adaptive Ising-based simulation flow, supporting both software-level experimentation and hardware-level validation.

The software folders are used to evaluate the proposed adaptive synthesis methodology, while the FPGA folder demonstrates the feasibility of mapping simplified probabilistic update behavior to digital hardware.

## Academic Context

This repository supports the development of a master's dissertation related to probabilistic processors based on the Ising model.

The project combines simulation, adaptive algorithm selection, benchmark evaluation, article result generation, and a hardware proof of concept using a low-cost FPGA board.

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

## Author

Jonathan Juracy Carneiro da Silva
Master's Student in Microelectronics
PGMICRO/UFRGS — Institute of Informatics
Federal University of Rio Grande do Sul
