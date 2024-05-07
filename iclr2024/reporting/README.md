# pyrmd

Python wrapper to create reports with [R Markdown](https://rmarkdown.rstudio.com/).

## Overview

This package is designed to let you produce PDF reports in Python with R Markdown. Given some data in Python, it will let you create a PDF that uses this data with one function call – `make_report`.

R Markdown is a tool to create documents that combine text, math, code, plots, and tables. R Markdown docs are similar to Jupyter notebooks in that they combine text with code. Unlike Jupyter notebooks, R Markdown documents can easily be converted into PDF/HTML/Word documents. This makes them a great tool to create PDF reports for experiments.

## Benefits

### Simplify Experiments

PDF reports make it easy to keep track of experiments and share results with collaborators. You can send a single PDF with all the plots, tables, parameters for an experiment, rather than 14 different files. Decluttering this process lets you focus on the fun parts of research.  

### Better Packages for Plots and Tables 

R Markdown can easily combine code in R and Python. For example, I run my experiments in Python, but produce the plots and tables with R packages like dplyr, ggplot2, and xtable. dplyr and ggplot2 make it much easier to create pretty plots than pandas and matplotlib. xtable lets you programmatically generate Latex tables from a data frame (there is no equivalent in Python).
   
### Source Files 

RMarkdown bundles each report with its source files. Each time you use RMarkdown to create a PDF report, it first produces TeX file for the report, then compiles the Tex file into a PDF. This Tex file is a godsend for research. If you're happy with the tables/plots for a specific experiment, you can cut and paste them into the paper.


## Setup

1. Install the latest version of [R](http://cran.wustl.edu/). 

2. Install the latest version of [RStudio](https://rstudio.com/products/rstudio/download/#download).
   
3. Install [MacTex](https://www.tug.org/mactex/) if you don't currently have a Tex package on your computer.

5. In RStudio, install the R packages for reporting with the snippet.

```
install.packages(pkgs = c("argparser", "reticulate", "rmarkdown", "knitr", "tidyverse", "xtable")
```

6. Update the Python interpreter for this project so that it has shared libraries enabled. One way to do this:
- install [PyEnv](https://opensource.com/article/19/5/python-3-default-mac);
- install a Python interpreter with shared libraries enabled (see [link](https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with---enable-shared))

Note: This is a painful step since you might have to setup a new environment. If you think that the Python interpreter already has shared libraries enabled (unlikely), you can skip this step. If you run into the `simpleError: Python shared library not found, Python bindings not loaded` bug, however, you will have to repeat this step.

To see the Python interpreter you are using in your current project, you can run `import sys; sys.executable`.

7. In a Python console, run the commands in the file `scripts/make_report.py` to create a sample PDF report from the template `demo_report.Rmd`. Running this file should output the file  `reports/iris_report.pdf` along with source files. 

### Known Issues

- `usage.py` fails with the message `simpleError: Python shared library not found, Python bindings not loaded`:  This means that the  your virtual environment for reticulate in (Step 6) is built from a python installation without shared libraries. The fix is to (1) install [PyEnv](https://opensource.com/article/19/5/python-3-default-mac); (2) install python with shared libraries enabled (see [link](https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with---enable-shared))

[Coming soon...]

## Learning More

If you're new to R, R Markdown, and reticulate are easy to pick up. The template `demo_report.Rmd` will show you the basics. The following guides should help you pick up the rest. 

- [R Markdown guide](https://bookdown.org/yihui/rmarkdown/) - Comprehensive guide on how to use R Markdown.

- [R Markdown reporting intro by Cosma Shalizi](https://www.stat.cmu.edu/~cshalizi/rmarkdown/) - this link contains a quick and dirty intro to R Markdown.

- [reticulate guide](https://rstudio.github.io/reticulate/) - This link contains a guide showing how you can call Python from R.