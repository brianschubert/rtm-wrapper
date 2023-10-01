RTM Wrapper
===========

|Code style: black| |Documentation Status|

Common wrapper interface for various Radiative Transfer Models (RTMs).

Currently supports `6S <https://salsa.umd.edu/6spage.html>`__.

Install
-------

With existing 6S installation:

.. code:: shell

   $ pip install .

With locally compiled 6S binary:

.. code:: shell

   $ pip install '.[6s]'

Without downloading dependencies:

.. code:: shell

   $ pip install --no-deps .

Test
----

.. code:: shell

   $ pytest

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-black.svg
   :target: https://github.com/psf/black
.. |Documentation Status| image:: https://readthedocs.org/projects/rtm-wrapper/badge/?version=develop
   :target: https://rtm-wrapper.readthedocs.io/en/develop/?badge=develop
