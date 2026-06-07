# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Source code path --------------------------------------------------------

sys.path.insert(0, os.path.abspath('../../src'))

# The following source code was created with AI assistance and has been human reviewed and edited.
# --
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Swectral'
copyright = '2025, Siwei Luo'
author = 'Siwei Luo'
release = '0.6.5'

# -- Sphinx extensions -------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',           # Automatically document Python code
    'sphinx.ext.autosummary',       # Generate summary tables
    'numpydoc',                     # Parse NumPy-style docstrings
    'sphinx_autodoc_typehints',     # Include type hints in docs
    'sphinx.ext.viewcode',          # Add "view source" links
    'sphinx.ext.intersphinx',       # Link to external docs
    'sphinx.ext.githubpages',       # Publish to GitHub Pages
    'sphinx.ext.autosectionlabel',  # Cross-reference sections
    'myst_parser',                  # Markdown support
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Numpydoc settings -------------------------------------------------------

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True  # Cross-reference types automatically
numpydoc_validate = True  # Warn on docstring errors

# -- Type hints --------------------------------------------------------------

autodoc_typehints = "none"
typehints_fully_qualified = False

# -- MyST settings -----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
]

# -- Autodoc / Autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_ignore_module_all = False  # Avoid duplication on class page

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'private-members': False,
    'undoc-members': False,
    'exclude-members': '__weakref__',
    'member-order': 'bysource',
}

# -- Templates ---------------------------------------------------------------

templates_path = ['_templates']

# -- Exclude patterns --------------------------------------------------------

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]

# -- Linkcheck configuration -------------------------------------------------
linkcheck_ignore = [
    r'https://siwei66\.github\.io/.*',
    r'https://doi\.org/.*',
    r'https://docs\.scipy\.org/.*',
    r'https://docs\.pytorch\.org/.*',
]
linkcheck_report_timeouts_as_broken = False

# -- HTML output -------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": True,
}

html_static_path = ['_static']
html_css_files = [
    "index.css",
]
html_show_sourcelink = False
html_favicon = "_static/favicon.ico"

# -- Intersphinx mapping -----------------------------------------------------
# Provides automatic cross-references to external docs for these libraries

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "PyWavelets": ("https://pywavelets.readthedocs.io/en/latest/", None),
}

# -- Options for sphinx-multiversion ---------------------------------------

# Whitelist which branches to build (only master)
smv_branch_whitelist = r'^master$'

# Whitelist which tags to build (matches tags like v1.0.0, v0.6.5)
smv_tag_whitelist = r'^v([0-9]|[1-9]\d+)\.(4|[5-9]|\d{2,})\.\d+$|^v[1-9]\d*\.\d+\.\d+$'

# Whitelist which remote to use
smv_remote_whitelist = r'^origin$'