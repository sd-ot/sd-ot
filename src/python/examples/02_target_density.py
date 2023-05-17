import pysdot as sdot
import numpy

# make a discretization of a gaussian function
t = numpy.linspace(-1,1,100)
x, y = numpy.meshgrid(t,t)
img = numpy.exp(-10 * (x**2 + y**2))

# find how to move mass
tm = sdot.find_transport_map(
    positions=numpy.random.rand(2000,2),
    domain=sdot.ScaledImage(min_pt=[0,0],max_pt=[1,1],img=img)
)
tm.show(arrows=True, line_width_arrows=2)


