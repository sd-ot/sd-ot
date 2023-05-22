Python tutorial
===============

.. _installation:

Installation
------------

To use Sdot, you can install it using pip:

.. code-block:: console

   $ pip install sdot

Optionally, if you want to work directly with the sources or with the latest version, you can install it manually

.. code-block:: console

   $ git clone https://github.com/sd-ot/sd-ot.git
   $ cd sd-ot
   $ make install # or `make link` to use symbolic links pointing to the source


A first transport map
---------------------

In this first example, we look for a transport map of a series of weighted diracs in a 0.2x0.2 square towards the characteristic function of the 1x1 square in 2D (the default domain in sdot).

.. code-block:: python

   import numpy, sdot

   # some constants
   nb_diracs = 5
   nb_dims = 2

   # optimize the kantorovitch potential
   tm = sdot.find_optimal_transport_map(
      sdot.make_weighted_diracs(
         numpy.random.rand(nb_diracs,nb_dims) * 0.2
      )
   )

   # visualization
   tm.show(arrows=True)

This gives a representation like:

.. image:: images/ex_arrow.png

where the arrows go from the dirac positions to the centroids of the corresponding cells (where the mass of each dirac goes).

`sdot.find_optimal_transport_map` returns a `TransportMap` object which contains methods for most common computations (e.g. `tm.transport_cost()` to get the overall transport cost). Depending on the input data, `sdot.find_optimal_transport_map` may return more specialized instances. In the example above, `tm` is an instance of `SemiDiscreteTransportMap` which contains methods like `tm.transport_cost_for_each_dirac()` or `tm.diagram()` (which gives a `PowerDiagram` for the L2 norm).


Another target density
----------------------

In the previous example, the target density was a simple characteristic function, but it is possible to define and use more complex ones. For instance, `sdot.make_uniform_grid_piecewise_polynomial` enable to define piecewise polynomial function on a regular grid

.. code-block:: python

   import numpy, sdot

   # make a discretization of a gaussian function using piecewise constant values
   t = numpy.linspace(-1,1,100)
   x, y = numpy.meshgrid(t,t)
   img = numpy.exp(-10 * (x**2 + y**2))

   # find how to move mass to the corresponding target density
   tm = sdot.find_optimal_transport_map(
      # source density
      sdot.make_weighted_diracs(
         numpy.random.rand(50,2)
      ),
      # target density
      sdot.make_uniform_grid_piecewise_polynomial(
         img, # value [x,y,n] where n is the number of coefficients of the polynomial
              # of 1, X, Y, X*X, X*Y, Y*Y, ... where X and Y equal 0 and 1 on the edges of pixels
         [0,0], # bottom left coordinates
         [1,1] # upper right coordinates
      )
   )

   tm.show(arrows=True, line_width_arrows=2)

This gives a representation like:

.. image:: images/ex_exp.png


An example in 3D
----------------

Here is the same problem in 3D:

.. code-block:: python

   import numpy, sdot

   t = numpy.linspace(-1,1,20)
   g = numpy.meshgrid(t,t,t)
   img = numpy.exp(-10 * sum(v**2 for v in g))

   tm = sdot.find_optimal_transport_map(
      sdot.make_weighted_diracs(
         numpy.random.rand(50,3)
      ),
      sdot.make_uniform_grid_piecewise_polynomial(
         img,
         [0,0,0],
         [1,1,1]
      )
   )

   # we write a vtk file to open it in paraview
   tm.write_vtk("ex.vtk")

.. image:: images/ex_3d.png


Using sdot objects
------------------

Most of the functions use objects instances to do the actual work. Using them directly may give access to some optimizations in term of computation time and code size.

In the following example, we make several computations that use the same source density. Using method calls enable `sdot` to keep track of the changes and cache the unmodified computations.


.. code-block:: python

   import numpy, sdot

   fo = sdot.OptimalTransportMapFinder(
      # source density
      sdot.make_weighted_diracs(
         numpy.random.rand(50,2)
      )
   )

   for num_iter in range(4):
      # target density
      fo.set_target_density(
         # here we use a symbolic expression
         sdot.make_symbolic_density(
            lambda x, y: - 10 ** num_iter * sdot.exp(x * x + y * y)
         )
      )

      # Computations that are specific to the source density are kept from each iteration to the next.
      # By default, the new Kantorovitch potentials are computed from those of the previous iteration.
      fo.run()

      # several output file to make an animation
      tm.write_vtk( f"ex_{ num_iter }.vtk" )



Large number of unknowns
------------------------




