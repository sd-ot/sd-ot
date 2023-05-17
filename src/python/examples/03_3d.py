import pysdot as sdot
import numpy

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


