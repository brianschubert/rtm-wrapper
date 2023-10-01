# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata

import rtm_wrapper.util

project = "RTM Wrapper"
author = " & ".join(importlib.metadata.metadata("rtm-wrapper").get_all("Author"))
copyright = f"2023, {author}"
release = rtm_wrapper.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "autoapi.extension"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# sphinx-autoapi configuration.
autoapi_dirs = ["../src/rtm_wrapper/"]
autoapi_options = [
    "members",
    # "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # "imported-members",
]

# Include type hints in both the signatures and description.
autodoc_typehints = "both"

# Include both class docstring and __init__ docstring.
autoapi_python_class_content = "both"

# Don't use fully qualified names when documenting module members.
add_module_names = False
