# reachml

[![python](https://img.shields.io/pypi/pyversions/reachml)](https://pypi.org/pypi/reachml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12820-b31b1b.svg)](https://arxiv.org/abs/2308.12820)
[![arXiv](https://img.shields.io/badge/arXiv-2410.22598-b31b1b.svg)](https://arxiv.org/abs/2410.22598)
[![arXiv](https://img.shields.io/badge/arXiv-2410.22598-b31b1b.svg)](https://arxiv.org/abs/2507.02169)
[![test](https://github.com/ustunb/reachml/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ustunb/reachml/actions/workflows/test.yml)

`reachml` is a library for recourse verification and statistical responsiveness testing.

## Background

*Recourse* is the ability of a decision subject to change the prediction of a machine learning model through actions on their features. *Recourse verification* aims to tell if a decision subject is assigned a prediction that is fixed. *

## Installation

You can install the library as follows:
```
pip install reachml
```

### CPLEX

The default solver in the package is [SCIP](https://www.scipopt.org), a source-available academic software with an open-source [Python API](https://github.com/scipopt/PySCIPOpt). If you want to use [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio), install `reachml` with:
```
pip install reachml[cplex]
```
The command above will install the [CPLEX Community Edition Python API](https://pypi.org/project/cplex/). The community edition has a strict limit on the size of models (e.g., number of constraints) it can support. To avoid these limits, you can purchase CPLEX and upgrade the Python API by running (on **CPLEX v22.1.2 and above**):
```
docplex config --upgrade PATH_TO_CPLEX_INSTALLATION    # e.g., ~/Applications/CPLEX_StudioXXXX
```

CPLEX Optimization Studio is free for students and academics through [IBM SkillsBuild](https://skillsbuild.org/) ([full instructions](https://community.ibm.com/community/user/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)).

## Quickstart Guide
The following example shows how to set up a dataset and run the thingie. This also outlines the steps follows in our `pipeline` script.

```python
import numpy as np
import pandas as pd

import reachml

# load dataset and create simple action set
X, y = reachml.datasets.example_shim()

# create action set
A = reachml.ActionSet(X)

#Immutable features
A[].actionable = False

# Initialize Constraints
#

# Create reachable sets
test = reachml.ReachableSet()
test.generate()

# Train model


# Initialize auditor

# Run audit
```

### Resources and Citations

For more about audit-based responsiveness scores, check out our 2025 paper: 

For more about feature-based responsiveness scores, check out our paper ICLR 2025 paper: [Feature Responsiveness Scores: Model-Agnostic Explanations for Recourse](https://arxiv.org/abs/2410.22598)

If you use responsiveness scores in your research, we would appreciate a citation:

```
@inproceedings{
    cheon2025feature,
    title={Feature Responsiveness Scores: Model-Agnostic Explanations for Recourse},
    author={Harry Cheon and Anneke Wernerfelt and Sorelle Friedler and Berk Ustun},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=wsWCVrH9dv}
}
```

The code for the paper is available under [research/iclr2025](https://github.com/ustunb/reachml/tree/main/research/iclr2025/).

For more about recourse verification, check out our paper ICLR 2024 spotlight paper: [Prediction without Preclusion](https://openreview.net/forum?id=SCQfYpdoGE)

If you use recourse verification in your research, we would appreciate a citation:
```
@inproceedings{kothari2024prediction,
    title={Prediction without Preclusion: Recourse Verification with Reachable Sets},
    author={Avni Kothari and Bogdan Kulynych and Tsui-Wei Weng and Berk Ustun},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=SCQfYpdoGE}
}
```

The code for the paper is available under [research/iclr2024](https://github.com/ustunb/reachml/tree/main/research/iclr2024/).