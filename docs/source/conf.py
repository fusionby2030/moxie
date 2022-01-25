# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Moxie'
copyright = '2021, fusionby2030'
author = 'fusionby2030, ajaervin'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_inherit_docstrings = False # This disables the pytorch and other documentation from appearing! 

autodoc_mock_imports = ['torch', 'pytorch-lightning', 'numpy', 'matplotlib', 'sklearn']
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
