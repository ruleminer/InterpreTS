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
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']
html_theme = 'sphinx_book_theme'
templates_path = ['_templates']
exclude_patterns = ['setup.py']
suppress_warnings = ["autodoc.import_object", "autodoc"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
