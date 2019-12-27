# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import codecs
import os
import re
import sys


# -- Project information -----------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META_PATH = os.path.join("src", "kerasadf", "__init__.py")

# add project root and package source to path
sys.path.insert(0, os.path.abspath(ROOT))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))


def read(*parts):
    """ Build an absolute path from *parts* and and return the contents of the
        resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(ROOT, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """ Extract __*meta*__ from META_FILE. """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


project = find_meta("title")
author = find_meta("author")
release = find_meta("version")
version = release.rsplit(u".", 1)[0]
copyright = find_meta("copyright")


# -- General configuration ---------------------------------------------------

# In nitpick mode (-n), still ignore any of the following "broken" references
# to non-types.
nitpick_ignore = [
    # ("py:class", "Any value"),
    # ("py:class", "callable"),
    # ("py:class", "callables"),
    # ("py:class", "iterable"),
    # ("py:class", "iterables"),
    # ("py:class", "tuple of types"),
    # ("py:class", "list of types"),
]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]
autosummary_generate = True

# numpydoc settings
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"type", "optional", "default", "of"}
numpydoc_xref_aliases = {
    "Constraint": "keras.constraints.Constraint",
    "Initializer": "keras.initializers.Initializer",
    "Regularizer": "keras.regularizers.Regularizer",
}
numpydoc_show_class_members = False  # only class names in toc
numpydoc_show_inherited_class_members = False  # only class names in toc
numpydoc_class_members_toctree = False  # we use a single page per class
autodoc_member_order = "groupwise"
autodoc_inherit_docstrings = False  # inherited strings not in numpydoc format

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/1.17/", None),
    "tensorflow": (
        "https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/",
        "objects_tf.inv",
    ),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "any"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if os.environ.get("READTHEDOCS"):
    html_theme = "sphinx_rtd_theme"
else:
    html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "{}-doc".format(project)

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, project, "{} Documentation".format(project), [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        project,
        "{} Documentation".format(project),
        author,
        project,
        find_meta("description"),
        "Miscellaneous",
    ),
]
