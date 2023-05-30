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

In this first example, we look for a 2D transport map from the characteristic function of the 1x1 square in 2D to a sum of weighted diracs in a 0.2x0.2 square, with the L2 norm for the transport cost.

.. code-block:: python

   import numpy, sdot

   # some constants
   nb_diracs = 5
   nb_dims = 2

   # optimize the kantorovitch potential
   tm = sdot.compute_optimal_transport_map(
      # source density
      #   Question pour Quentin: on pourrait peut être utiliser `characteristic_function`
      #   pour rester dans l'ambiance mais ça serait peut être pompeux...
      sdot.is_inside(
         # sdot contains some basic geometry tools
         sdot.parallelogram(
            # first corner
            [0, 0],
            # axes
            [[1, 0], [0, 1]]
         )
      ),
      # target density
      sdot.weighted_point_cloud(
         # positions
         numpy.random.rand(nb_diracs, nb_dims) * 0.2,
         # weights (so that masses of the source and target dentities are the same)
         numpy.ones(nb_diracs) / nb_diracs
      )
   )

   # visualization (arrows are possible because at least one density is a sum of diracs)
   tm.show(arrows=True)

This gives a representation like:

.. image:: images/ex_arrow.png

where the arrows go from the centroids of the cells (subspace assigned to a given dirac) the dirac positions.

`sdot.compute_optimal_transport_map` returns a `TransportMap` object which contains methods for the most common computations (e.g. `tm.transport_cost()` to get the overall transport cost). Depending on the input data, `sdot.compute_optimal_transport_map` may return more specialized instances.

In the example above, `tm` is actually an instance of `SemiDiscreteTransportMap` which contains methods like `tm.transport_cost_for_each_dirac()` or `tm.diagram()` which gives a `PowerDiagram` (because or the L2 norm) which contains a bunch of geometrical computation methods.


More complex densities
----------------------

In general, densities are handled in sdot as symbolic formula.

It is possible to construct them using helper functions as seen before (`sdot.characteristic_function`, `sdot.weighted_point_cloud`) or starting with coordinate symbols (`sdot.coords`) with generic construction operators (`+`, `sdot.sum`, ...).

In the following example, one will use a bounded gaussian function:

.. code-block:: python

   import numpy, sdot

   # symbolic formula for the source density.
   source = sdot.exp(- 10 * sdot.sum(- sdot.coords ** 2)) * \
            # Rq/question pour Quentin: j'imagine qu'on pourrait traiter des domaines non bornés,
            # ce qui simplifierait l'exemple, mais nous emmèrait peut-être sur des développements
            # conséquents et peut-être inutiles (je suis pas certain de pouvoir estimer ça rapidement)
            sdot.is_inside(sdot.parallelogram([-1, -1], [[2, 0], [0, 2]]))

   # mass of the source to get the correct mass for the target density
   target = sdot.weighted_point_cloud(
      # positions
      numpy.random.rand(nb_diracs, nb_dims) * 2 - 1,
      # weights (so that masses of the source and target dentities are the same)
      numpy.ones(nb_diracs) / nb_diracs * sdot.mass(source)
   )

   # compute and display
   tm = sdot.compute_optimal_transport_map(source, target)
   tm.show(arrows=True, line_width_arrows=2)

This gives a representation like:

.. image:: images/ex_exp.png


Here is another example where the source density is defined piecewise on an uniform grid:

.. code-block:: python

   import numpy, sdot, PIL.Image

   # one loads an image and makes sure that the mass is equal to 1
   img = np.asarray(PIL.Image.open("ot.png"))
   img = np.sum(img * 1.0 + 25, axis=2)
   img = img / np.mean(img)

   # find how to move mass to the corresponding target density
   tm = sdot.compute_optimal_transport_map(
      sdot.piecewise_constant_on_an_uniform_grid(
         img, # values
         [0, 0], # first corner
         [[1, 0], [0, 1]] # axes
      ),
      sdot.weighted_point_cloud(
         numpy.random.rand(50, 2)
      )
   )

   tm.show(arrows=True, line_width_arrows=2)

.. image:: images/ex_img.png


An example in 3D
----------------

Changing

Here is the same problem in 3D:

.. code-block:: python

   import numpy, sdot

   t = numpy.linspace(-1,1,20)
   g = numpy.meshgrid(t,t,t)
   img = numpy.exp(-10 * sum(v**2 for v in g))

   tm = sdot.compute_optimal_transport_map(
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



Other transport costs
---------------------

Unbalanced



Large number of unknowns
------------------------




