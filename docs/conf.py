# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# docs/conf.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> /Users/meredith/reachml
sys.path.insert(0, str(ROOT))

# Optional if you have heavy/optional deps not installed:
autodoc_mock_imports = [
    "cplex", "docplex", "pyscipopt",
    "scipy", "numpy", "sklearn",    # ← add these
    "rich", "pandas", "h5py"              # (add any others your code imports at module import time)
]



project = 'reachml'
copyright = '2025, "Harry Cheon, Avni Kothari, Bogdan Kulynych, Meredith Stewart, Berk Ustun"'
author = '"Harry Cheon, Avni Kothari, Bogdan Kulynych, Meredith Stewart, Berk Ustun"'
release = '"1.0"'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # supports Google/Numpy style docstrings
    'sphinx.ext.viewcode',  # adds “View Source” links
    'sphinx.ext.autosummary',
    'myst_parser',          # optional for Markdown
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
