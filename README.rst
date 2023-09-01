.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/sphere-n.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/sphere-n
    .. image:: https://readthedocs.org/projects/sphere-n/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://sphere-n.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/sphere-n/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/sphere-n
    .. image:: https://img.shields.io/pypi/v/sphere-n.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/sphere-n/
    .. image:: https://img.shields.io/conda/vn/conda-forge/sphere-n.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/sphere-n
    .. image:: https://pepy.tech/badge/sphere-n/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/sphere-n
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/sphere-n

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/
.. image:: https://readthedocs.org/projects/sphere-n/badge/?version=latest
    :target: https://sphere-n.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/gh/luk036/sphere-n/branch/main/graph/badge.svg?token=86zuv3iw31
    :target: https://codecov.io/gh/luk036/sphere-n

======
sphere-n
======


    A library for low-discrepancy sequences

The desirable properties of samples over n-sphere include being uniform, deterministic, and incremental. The uniformity measures are optimized with every new point, and this is because in some applications, it is unknown how many points are needed to solve the problem in advance.

Some potential applications of generating samples over n-sphere for n > 2 include robotic motion planning, spherical coding in MIMO wireless communication, cookbook for unitary matrices, multivariate empirical mode decomposition, and filter bank design.

Numerical results show that the proposed method outperforms the randomly generated sequences and other proposed methods.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
