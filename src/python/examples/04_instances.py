from pysdot.domain_types import ScaledImage
from pysdot import OptimalTransport
import numpy as np

import pyvista as pv
from pyvista import themes
pv.set_plot_theme("document")
# pv.set_plot_theme(themes.document())
pv.global_theme.anti_aliasing = 'ssaa'
# pv.set_plot_theme(themes.ParaViewTheme())
# pv.global_theme.depth_peeling.enabled = True

# initial positions
n = 500
positions = np.random.rand(n,2)

ot = OptimalTransport(positions)
ot.verbosity = 2

# solve
for l in range(8):
    t = np.linspace(-1,1,100)
    x, y = np.meshgrid(t,t)
    img = np.exp( - 2**l * (x**2 + y**2) )
    img /= np.mean(img)
    
    ot.set_domain(ScaledImage([0, 0], [1, 1], img))
    ot.adjust_weights()

    # ot.pd.display_vtk( f"results/pd_{ l }.vtk" )
    cells, celltypes, points = ot.pd.vtk_mesh_data(0.001)
    m = pv.UnstructuredGrid(cells, celltypes, points) # .shrink(0.999)


    p = pv.Plotter()

    mi = np.min( ot.pd.positions, axis = 0 )
    ma = np.max( ot.pd.positions, axis = 0 )
    me = ( mi + ma ) / 2
    di = ma - mi
    p.set_focus([me[0], me[1], 0])
    p.set_position([me[0], me[1], 2 * np.max(di)])
    p.set_viewup([0, 1, 0])

    # p.clear()
    p.add_mesh(m, show_edges=True, line_width=3, style="wireframe", color="black")
    p.show(interactive=False, screenshot=f"results/pd_{ l }.png", window_size=[800,800])
    # p.screenshot( f"results/pd_{ l }.png", transparent_background=True )