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

.. warning::
   Pour Quentin: parler d'une densité est peut être une mauvaise idée dans la mesure où la masse n'égale pas toujours 1 dans nos exemples. Est-ce qu'on aurait un mot plus approprié ?


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

      # several output files, to make an animation
      tm.write_vtk( f"ex_{ num_iter }.vtk" )



Transport cost
--------------

By default, sdot uses the L2 norm for the transport cost (:math:`\int ||x - y||^2_2 d\rho`). It is possible to define another transport costs using names (e.g. "L2", ...) or functions.

Functions to define transport costs take an input argument that contains several attributes: `x` is the position of a source item, `y` is the position of a target item, `w` is the kantorovitch potential and `m` is the created mass (which will be 0 if not used). Additionally, there are shortcuts, like `d_2` for instance which equals :math:`||x - y||_2`.

.. warning::
   Pour Quentin: "item" n'est peut-être pas le meilleur terme mais je n'ai pas su quoi mettre...

Here is an example where the cost becomes infinite if the square of the distance is greater than the Kantorovitch potential.

.. code-block:: python

   import numpy, sdot

   target_radius = 0.05
   nb_diracs = 100

   tm = sdot.find_optimal_transport_map(
      sdot.make_weighted_diracs( 
         numpy.random.rand(nb_diracs, 2),
         # here we specify the mass of each dirac individually
         np.ones(nb_diracs) * np.pi * target_radius ** 2
      ),
      # we use sdot objects to make a symbolic function that will be compiled
      # to produce an optimized code
      transport_cost=lambda p: p.d_2 ** 2 + sdot.inf * (p.d_2 ** 2 > p.w),
   )

   tm.show()

It produces something like:

.. image:: images/ex_r2.png


Here is an example with unbalanced mass tranport to illustrate the use of the `m` attribute:

.. code-block:: python

   import numpy, sdot

   nb_diracs = 100

   tm = sdot.find_optimal_transport_map(
      sdot.make_weighted_diracs( 
         numpy.random.rand(nb_diracs,2),
         # the mass of the source distribution is not equal to the mass of the target distribution
         np.ones(nb_diracs) / nb_diracs
      ),
      # target distribution
      sdot.make_symbolic_density(
         lambda p: - sdot.exp(sdot.norm_2(p) ** 2)
      )
      # creation or destruction of the mass is allowed in this question
      transport_cost=lambda p: p.d_2 ** 2 + 10 * p.m,
   )

   tm.show()

.. warning::
   Pour Quentin: cette exemple ne fonctionne pas encore et je ne suis même pas certain qu'on soit sur la bonne formule pour le coût. À discuter.


Large number of unknowns
------------------------

To handle things like MPI calls, out-of-core data, ... sdot tries to be as flexible as possible, notably in terms of framework choice. Currently, we support Dask and CuPy but if one needs to use sdot with another libraries we will be happy to develop it.

Here is an example with data specified with Dask:


.. code-block:: python

   from dask.distributed import Client
   import dask.array as da
   import numpy, sdot

   client = Client(n_workers=4)

   # here we take a dask array as input
   tm = sdot.find_optimal_transport_map(
      sdot.make_weighted_diracs( 
         da.random.rand((1000000,2), chunks=4)
      ),
   )

   # transport_cost_for_each_dirac will return a Dask array
   print(da.sum(tm.transport_cost_for_each_dirac()))


