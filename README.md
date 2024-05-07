# reachml

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12820-b31b1b.svg)](https://arxiv.org/abs/2308.12820)
[![CI](https://github.com/ustunb/reachml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ustunb/reachml/actions/workflows/ci.yml)


Library for recourse verifictaion and the accompanying code for the paper "[Prediction without Preclusion: Recourse Verification with Reachable Sets](https://arxiv.org/abs/2308.12820)" (ICLR 2024 Spotlight).

## Getting Started

### Installing the Library
Install as:
```
pip install git+https://github.com/ustunb/reachml#egg=reachml
```
The library relies on [IBM CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). The free version of CPLEX is significantly slower than the commercial/academic one. To make it fast, you want to additionally install the closed-source version of Cplex in the virtual environment of the package.
