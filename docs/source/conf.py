# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'InterpreTS'
copyright = '2024, Łukasz Wróbel, Sławomir Put, Martyna Żur, Martyna Kramarz, Jarosław Strzelczyk, Weronika Wołowczyk, Piotr Krupiński'
author = 'Łukasz Wróbel, Sławomir Put, Martyna Żur, Martyna Kramarz, Jarosław Strzelczyk, Weronika Wołowczyk, Piotr Krupiński'
release = '0.5.0'
html_logo = None

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'myst_parser',
              'nbsphinx']
html_theme = 'pydata_sphinx_theme'
templates_path = ['_templates']
exclude_patterns = ['setup.py']
suppress_warnings = ["autodoc.import_object", "autodoc"]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
    'private-members': False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
