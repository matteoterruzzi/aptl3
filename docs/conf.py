# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'aptl3'
copyright = '2021, Matteo Terruzzi'
author = 'Matteo Terruzzi'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.inheritance_diagram',  # NOTE: requires Graphviz to be installed
    'sphinx.ext.autosummary',
    'autoapi.extension',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'venv']

default_role = 'py:obj'

inheritance_graph_attrs = dict(
    rankdir="BT", ranksep=0.3, margin=0.2, ratio='compress')
inheritance_edge_attrs = dict(
    arrowtail='onormal', dir='back')


# -- autoapi configuration ---------------------------------------------------

autoapi_generate_api_docs = True
autoapi_add_toctree_entry = True
autoapi_root = 'autoapi'  # WARNING: contents of the directory are deleted!
autoapi_dirs = ['../aptl3/']
autoapi_ignore = [
    '*/aptl3/qt/*',
    '*/aptl3/scripts/*',
    '*/aptl3/__main__.py',
    '*/aptl3/tests/*',
]

autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-inheritance-diagram',  # NOTE: requires Graphviz to be installed
    'show-module-summary',
    'special-members',
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
