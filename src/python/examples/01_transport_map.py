import pysdot as sdot
import numpy

# some constants
nb_diracs = 5
nb_dims = 2

# optimize the kantorovitch potential
tm = sdot.find_transport_map(
    positions=numpy.random.rand(nb_diracs,nb_dims) * 0.2 + 0.01,
    masses=numpy.ones([nb_diracs])/nb_diracs # so that mass(diracs) = mass(domain)
)

# visualization
tm.show(arrows=True)


