Python tutorial
===============

.. _installation:

Installation
------------

To use Sdot, you can install it using pip:

.. code-block:: console

   $ pip install sdot

Optionally, if you want to work directly with the source, for instance to make change or get the latest version, 

A simple problem
----------------

In this first example, we look for a transport map of a series of weighted diracs towards the characteristic function of the unit square in 2D (the default domain).

.. code-block:: python

   import sdot, numpy

   # some constants
   nb_diracs = 10
   nb_dims = 2

   # optimize the kantorovitch potential
   tm = sdot.find_transport_map(
      positions=numpy.random.rand(nb_diracs,nb_dims),
      masses=numpy.ones([nb_diracs])/nb_diracs # so that mass(diracs) = mass(domain)
   )

   # visualization
   tm.show(arrows=True)


