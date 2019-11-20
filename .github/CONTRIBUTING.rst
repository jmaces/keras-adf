How To Contribute
=================

Thank you for considering contributing to ``keras-adf``!
Everyone is very welcome to help us improve it.

This document is intended to help you get started and make the process of
contributing more accessible. Do not be afraid ask if something is unclear!


Workflow
--------

- Every contribution is welcome, no matter how small!
  Do not hesitate to submit fixes for typos etc.
- Try to stick to *one* change only per pull request.
- Add tests and docs for your code. Contributions missing tests or
  documentation can not be merged.
- Make sure all changes pass our CI_.
  We will not give any feedback until it is green unless you ask for it.
- Once you have addressed review feedback bump the pull request with a short note, so we know you are done.


Code
----

- We follow `PEP 8`_ and `PEP 257`_.
- We use `numpy style`_ for docstrings:

    .. code-block:: python

      def func(x):
        """Short summary.

        Longer description text.

        Parameters
        ----------
        param1 : int
            The first parameter.
        param2 : str
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
- If you make additions or changes to the public APIs, tag the docstring with
  ``..  versionadded:: 19.5.0 NOTE`` or ``..  versionchanged:: 19.7.0 NOTE``.
- We use isort_ to sort all imports, and follow the Black_ code style with a
  line length of 79 characters. The formatting can be automated. If you run our
  tox test suite before committing, or install our pre-commit_ hooks (see below),
  you do not have to spend any thoughts or time on formatting your code at all.
  Otherwise CI will catch it, but then you may waste time before getting the
  green light.


Tests
-----

- Write assertions as ``expected == actual`` for consistency:

  .. code-block:: python

     x = f(...)

     assert 42 == x.important_attribute
     assert "foobar" == x.another_attribute

- Get the latest version of tox_ to run our tests with one single ``tox`` call.
  It will ensure the test suite runs with all the correct dependencies against
  all supported Python versions just as it will in our CI.
  If you lack Python versions, you can can limit the environments like
  ``tox -e py27,py35``.
- Write docstrings for your tests. Here are tips for writing `good test docstrings`_.


Documentation
-------------

- Use `semantic newlines`_ in reStructuredText_ files (files ending in ``.rst``):

  .. code-block:: rst

     This is a sentence.
     This is another sentence.

- If you start a new section, add two blank lines before and one blank line after the header, except if two headers follow immediately after each other:

  .. code-block:: rst

     Last line of previous section.


     Header of New Top Section
     -------------------------

     Header of New Section
     ^^^^^^^^^^^^^^^^^^^^^

     First line of new section.

- If you add a new feature, demonstrate it on the `examples page`_!


Changelog
^^^^^^^^^

If you make a change noteworthy for all users, there needs to be a changelog
entry to make everyone else aware about it!

We use the towncrier_ package to manage our changelog.
``towncrier`` uses independent files -- called *fragments* -- for each pull
request instead of one monolithic changelog file. On release, all fragments
are compiled into the ``CHANGELOG.rst``.

You don't need to install ``towncrier`` yourself, since you will not be the
one releasing a new version. You just have to abide by a few simple rules:

- For each pull request, add a new file into ``changelog.d`` with a filename
  adhering to the ``pr#.(change|deprecation|breaking).rst`` schema:
  For example, ``changelog.d/42.change.rst`` for a non-breaking change that is
  proposed in pull request #42.
- As with other docs, please use `semantic newlines`_ within news fragments.
- Wrap symbols like modules, functions, or classes into double backticks so
  they are rendered in a ``monospace font``.
- Wrap arguments into asterisks like in docstrings.
- If you mention functions or other callables, add parentheses at the end of
  their names for readability: ``func()`` or ``object.method()``.
- Prefer simple past tense or constructions with "now". For example:

  + Added ``func()``.
  + ``func()`` now does not crash with argument 42.
- If you want to reference multiple issues, copy the fragment content to
  another filename. ``towncrier`` merges all fragments with identical contents
  into one entry with multiple pull request links.

----

``tox -e changelog`` will render the current changelog to the terminal if you
want to double check your fragments.


Local Development Environment
-----------------------------

You can (and should) run our test suite using tox_.
For a more traditional environment we recommend to develop using the latest
Python 3 release.

Create a new virtual environment using your favourite environment manager.
Then get an up to date checkout of the ``keras-adf`` repository:

.. code-block:: bash

    $ git clone git@github.com:jmaces/keras-adf.git

or if you want to use git via ``https``:

.. code-block:: bash

    $ git clone https://github.com/jmaces/keras-adf.git

Change into the newly created directory and **activate your virtual environment**
if you have not done that already. Install an editable version of ``keras-adf``
along with all its development requirements:

.. code-block:: bash

    $ cd keras-adf
    $ pip install -e '.[dev]'

Now you should be able to run tests via

.. code-block:: bash

   $ python -m pytest

as well as building documentation via

.. code-block:: bash

   $ cd docs
   $ make html

which can then be found in ``docs/_build/html/``.

To avoid committing code not following our style guide, we advise you to
install pre-commit_ [#f1]_ hooks:

.. code-block:: bash

   $ pre-commit install

They can also be run manually anytime (as our tox does) using:

.. code-block:: bash

   $ pre-commit run --all-files


.. [#f1] pre-commit should have been installed into your virtual environment automatically
         when you ran ``pip install -e '.[dev]'`` above. If pre-commit is
         missing, you may need to re-run ``pip install -e '.[dev]'``.


Governance
----------

``keras-adf`` was created as a byproduct of a research project and is
maintained by volunteers. We are always open to new members that want to help.
Just let us know if you want to join the team.

**Everyone is welcome to help review/merge pull requests of others but nobody
is allowed to merge their own code.**

`Jan Maces`_ acts as the maintainer of the project and has the final say over decisions.


****

Please note that this project is released with a Contributor `Code of Conduct`_.
By participating in this project you agree to abide by its terms.
Please report any harm to `Jan Maces`_ in any way you find appropriate.

Thank you again for considering contributing to ``keras-adf``!


.. _`Jan Maces`: https://github.com/jmaces
.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
.. _`PEP 257`: https://www.python.org/dev/peps/pep-0257/
.. _`good test docstrings`: https://jml.io/pages/test-docstrings.html
.. _`Code of Conduct`: https://github.com/jmaces/keras-adf/blob/master/.github/CODE_OF_CONDUCT.rst
.. _changelog: https://github.com/jmaces/keras-adf/blob/master/CHANGELOG.rst
.. _tox: https://tox.readthedocs.io/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _semantic newlines: https://rhodesmill.org/brandon/2012/one-sentence-per-line/
.. _examples page: https://github.com/jmaces/keras-adf/blob/master/docs/examples.rst
.. _CI: https://travis-ci.com/jmaces/keras-adf
.. _towncrier: https://pypi.org/project/towncrier
.. _black: https://github.com/psf/black
.. _pre-commit: https://pre-commit.com/
.. _isort: https://github.com/timothycrosley/isort
.. _`numpy style`: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
