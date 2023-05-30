Welcome to Sdot's documentation!
================================

**Sdot** is a set of tools to help solve problems related to semi-discrete optimal transportation.

These tools have been designed for speed (see :doc:`benchmarks`) and robustness. They work for any (reasonable) number of dimension.

The core is written in C++ and Cuda (optionnally) and there are wrappers for most of the common scientific programming languages. 

If you're new to semi-discrete optimal transport see :doc:`sd-intro` for some generic mathematical explanations and :doc:`showcases` to get a taste of what can be done using this kind of tools.

Check out the :doc:`tuto_py`, :doc:`tuto_jl` or :doc:`tuto_cpp` to start coding some examples in your favorite language.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   sd-intro
   tuto_py
   tuto_jl
   tuto_cpp
   showcases
   benchmarks