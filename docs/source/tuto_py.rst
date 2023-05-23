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
      sdot.dirac_distribution(
         numpy.random.rand(nb_diracs,nb_dims) * 0.2
      )
   )

   # visualization
   tm.show(arrows=True)

This gives a representation like:

.. image:: images/ex_arrow.png
   :width: 300
   :align: center

where the arrows go from the dirac positions to the centroids of the corresponding cells (where the mass of each dirac goes).

`sdot.find_optimal_transport_map` actually returns a `TransportMap` object which contains methods for most common computations (e.g. `tm.transport_cost()` to get the overall transport cost, ...).

Depending on the input data, `sdot.find_optimal_transport_map` may return more specialized instances. In the example above, `tm` is an instance of `SemiDiscreteTransportMap` which contains methods like `tm.transport_cost_for_each_dirac()` or `tm.diagram()` which gives an instance of a `PowerDiagram`.


Target densities
----------------

.. warning::
   Pour Quentin: parler d'une densité est peut être une mauvaise idée dans la mesure où la masse dans nos exemples n'est pas toujours égale à 1. Est-ce que "distribution" serait ok ?


Symbolic expressions allow to define more complex source or target densities.

These symbolic expressions can be constructed directly using symbols like `sdot.coords` (which gives the space coordinates) and helpers like for instance `sdot.bounded` (that multiplies an expression by 0 is outside of a given geometry, by default the unit square/cube/...).

Additionaly, Sdot gives more specialized helper functions, like `sdot.dirac_distribution` as seen before. For instance `sdot.uniform_grid_piecewise_polynomial_distribution` enables to define a piecewise polynomial function on an uniform regular grid:

.. code-block:: python

   import numpy, sdot

   # make a discretization of a gaussian function using piecewise constant values (polynomial order=0)
   t = numpy.linspace(-1,1,100)
   x, y = numpy.meshgrid(t,t)
   img = numpy.exp(-10 * (x**2 + y**2))

   # find how to move mass from diracs to a piecewise function
   tm = sdot.find_optimal_transport_map(
      # source density
      sdot.dirac_distribution(
         numpy.random.rand(50,2)
      ),
      # target density
      sdot.uniform_grid_piecewise_polynomial_distribution(
         img, # value [x,y,n] where n is the number of coefficients of the polynomial
              # of 1, X, Y, X*X, X*Y, Y*Y, ... where X and Y equal 0 and 1 on the edges of pixels
         [0,0], # bottom left coordinates
         [1,1] # upper right coordinates
      )
   )

   tm.show(arrows=True, line_width_arrows=2)

This gives a representation like:

.. image:: images/ex_exp.png
   :width: 300
   :align: center


Space dimension
---------------

Sdot tries to find the dimension according to the input data and the API globally stays the same. This is an example of a 3D computation:

.. code-block:: python

   import numpy, sdot

   t = numpy.linspace(-1,1,20)
   g = numpy.meshgrid(t,t,t)
   img = numpy.exp(-10 * sum(v**2 for v in g))

   tm = sdot.find_optimal_transport_map(
      sdot.dirac_distribution(
         numpy.random.rand(50,3)
      ),
      sdot.uniform_grid_piecewise_polynomial_distribution(
         img,
         [0,0,0],
         [1,1,1]
      )
      # dim=... to specify the space dimension
   )

   # we write a vtk file to open it in paraview
   tm.write_vtk("ex.vtk")

.. image:: images/ex_3d.png
   :width: 300
   :align: center


Using sdot objects
------------------

Most of the functions use instances of Sdot objects to do the actual work. Using them directly may give access to some optimizations, both in term of computation time and code size.

In the following example, we compute several transport map that use the same source density. Using instances allows Sdot to cache the unmodified computations and use previous ones as starting points.


.. code-block:: python

   import numpy, sdot

   fo = sdot.OptimalTransportMapFinder(
      sdot.dirac_distribution(
         numpy.random.rand(50,2)
      )
   )

   for num_iter in range(4):
      # mod/set of the target density
      fo.set_target_density(
         # here we use a symbolic expression
         sdot.bounded(- 10 ** num_iter * sdot.exp(sdot.sum(sdot.coords ** 2)))
      )

      # Computations that are specific to the source density are kept from each iteration to the next.
      # By default, the new Kantorovitch potentials are computed from those of the previous iteration.
      fo.run()

      # animation
      tm.write_vtk( f"ex_{ num_iter }.vtk" )


.. image:: images/ex_inst.gif
   :width: 300
   :align: center


Transport cost
--------------

By default, sdot uses the L2 norm for the transport cost (:math:`\int ||x - y||^2_2 d\rho`). Of course, it is possible to define another transport costs. It can be done using names for the most common ones (e.g. "L2", ...) or symbolic expression to get more flexibility.

Expressions may use the following symbol: `sdot.source_pos` is the position of a source item, `sdot.target_pos` is the position of a target item, `sdot.kantorovitch_potential` is the kantorovitch potential and `sdot.created_mass` is the created mass (which will be 0 if not used in the cost expression). Additionally, there are shortcuts, like for instance `sdot.distance_2` which is the norm 2 of the distance between `sdot.source_pos` and `sdot.target_pos`.

.. warning::
   Pour Quentin: "item" n'est peut-être pas le meilleur terme mais je n'ai pas su quoi mettre...

Here is an example where the cost becomes infinite if the square of the distance is greater than the Kantorovitch potential.

.. code-block:: python

   import numpy, sdot

   target_radius = 0.05
   nb_diracs = 100

   tm = sdot.find_optimal_transport_map(
      sdot.dirac_distribution( 
         numpy.random.rand(nb_diracs, 2),
         # for this example we specify the mass of each dirac individually
         np.ones(nb_diracs) * np.pi * target_radius ** 2
      ),
      transport_cost=sdot.distance_2 ** 2 + sdot.inf * (sdot.distance_2 ** 2 > sdot.kantorovitch_potential),
   )

   tm.show()

It produces something like:

.. image:: images/ex_r2.png
   :width: 300
   :align: center


Here is an example with unbalanced mass tranport to illustrate the use of the `sdot.created_mass` symbol:

.. code-block:: python

   import numpy, sdot

   nb_diracs = 100

   tm = sdot.find_optimal_transport_map(
      sdot.dirac_distribution( 
         numpy.random.rand(nb_diracs,2),
         # the mass of the source distribution is not equal to the mass of the target distribution
         np.ones(nb_diracs) / nb_diracs
      ),
      # target distribution
      sdot.exp(- sdot.norm_2(sdot.coords) ** 2)
      # creation or destruction of the mass is allowed in this example
      transport_cost=p.distance_2 ** 2 + 10 * p.created_mass,
   )

   tm.show()

.. warning::
   Pour Quentin: cette exemple ne fonctionne pas encore et je ne suis même pas certain qu'on soit sur le bon genre de formule pour le coût. À discuter.


Large number of unknowns
------------------------

To handle things like MPI calls, out-of-core data, GPUs, ... sdot tries to be as flexible as possible, notably in terms of framework choice.

Currently, for python, we support Dask and CuPy but if one needs to use sdot with another libraries we will be happy to develop the interfaces.

Here is an example with data specified with Dask:


.. code-block:: python

   from dask.distributed import Client
   import dask.array as da
   import numpy, sdot

   client = Client(n_workers=4)

   # here we take a dask array as input
   tm = sdot.find_optimal_transport_map(
      sdot.dirac_distribution( 
         da.random.rand((1000000,2), chunks=4)
      ),
   )

   # in this case, transport_cost_for_each_dirac will return a Dask array
   print(da.sum(tm.transport_cost_for_each_dirac()))


