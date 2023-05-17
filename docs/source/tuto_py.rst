Python tutorial
===============

.. _installation:

Installation
------------

To use Sdot, you can install it using pip:

.. code-block:: console

   $ pip install sdot

Optionally, if you want to work directly with the source, for instance to make change or get the latest version, 

A first transport map
---------------------

In this first example, we look for a transport map of a series of weighted diracs in a 0.2x0.2 square towards the characteristic function of the 1x1 square in 2D (the default domain in sdot).

.. code-block:: python

   import numpy, sdot

   # some constants
   nb_diracs = 5
   nb_dims = 2

   # optimize the kantorovitch potential
   tm = sdot.find_transport_map(
      positions=numpy.random.rand(nb_diracs,nb_dims) * 0.2,
   )

   # visualization
   tm.show(arrows=True)

This gives a representation like:

.. image:: images/ex_arrow.png

where the arrows go from the dirac positions to the centroids of the corresponding cells (where the mass of each dirac goes).

Target densities
----------------

In the previous example, the target density was a simple characteristic function, but it is possible to define and use more complex ones. For instance, `sdot.ScaledImage` enable to define piecewise polynomial function

.. code-block:: python

   import numpy, sdot

   # make a discretization of a gaussian function using piecewise constant values
   t = numpy.linspace(-1,1,100)
   x, y = numpy.meshgrid(t,t)
   img = numpy.exp(-10 * (x**2 + y**2)) # add extra dimensions for higher degrees

   # find how to move mass to the corresponding target density
   tm = sdot.find_transport_map(
      positions=numpy.random.rand(50,2),
      domain=sdot.ScaledImage(min_pt=[0,0],max_pt=[1,1],img=img)
   )
   tm.show(arrows=True, line_width_arrows=2)

This gives a representation like:

.. image:: images/ex_exp.png


Higher dimensions
-----------------

Changing the number of dimensions of the problem does not change the calls to be made:

.. code-block:: python

   import numpy, sdot

   t = numpy.linspace(-1,1,20)
   g = numpy.meshgrid(t,t,t)
   img = numpy.exp(-10 * sum(v**2 for v in g))
   print( img )

   tm = sdot.find_transport_map(
      positions=numpy.random.rand(50,3),
      domain=sdot.ScaledImage(min_pt=[0,0,0],max_pt=[1,1,1],img=img)
   )

   # we write a vtk file to open it in paraview
   tm.write_vtk("ex.vtk")

.. image:: images/ex_3d.png
