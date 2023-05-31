from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
import pylab as plt
import numpy as np
import timeit
import scipy

def test_egan():
    T = 1
    X = scipy.io.loadmat( "test_case.mat" )[ "Z_test" ]
    positions = X[ :, :, T ]
    nb_diracs = positions.shape[ 0 ]
    # plt.plot( X[ :, 0, 2 ], X[ :, 1, 2 ], '.' )
    # plt.show()

    print( nb_diracs )

    domain = ConvexPolyhedraAssembly()
    domain.add_box( np.min( positions, axis=0 ), np.max( positions, axis=0 ) )

    ot = OptimalTransport( positions, domain = domain, masses = np.ones( nb_diracs ) / nb_diracs * domain.measure(), obj_max_dw = 10 )
    ot.verbosity = 2

    ot.adjust_weights( relax = 0.5 )

    ot.display_vtk( "results/pd.vtk" )

def test_aniso( ratio = 1e-2 ):
    positions = np.random.rand( 200000, 2 ) # * np.array([ 1, 1e-2 ])
    nb_diracs = positions.shape[ 0 ]

    domain = ConvexPolyhedraAssembly()
    domain.add_box( np.min( positions, axis=0 ), np.max( positions, axis=0 ) )

    ot = OptimalTransport( positions, domain = domain, masses = np.ones( nb_diracs ) / nb_diracs * domain.measure(), obj_max_dw = 1e-6 )
    ot.verbosity = 2

    # mvs = self.pd.der_integrals_wrt_weights(stop_if_void=True)
    print( timeit.timeit( ot.pd.der_integrals_wrt_weights, number=1 ) )

    # ot.adjust_weights()
    # ot.display_vtk( "results/pd.vtk" )

test_aniso()
