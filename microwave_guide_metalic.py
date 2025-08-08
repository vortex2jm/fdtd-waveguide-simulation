import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parameters ------------------
Size_x = 40   # cm
Size_y = 40   # cm
Size_z = 0
cell_size = mp.Vector3(Size_x, Size_y, Size_z)

resolution = 5  # cells per cm
pml_size = 2  # cm (10 cells with resolution 5)
pml_layers = [mp.PML(thickness=pml_size)]

# ------------------ Medium ------------------
air = mp.Medium(epsilon=1.0)
metal = mp.Medium(D_conductivity=1e7)  # High conductivity

# ------------------ Waveguide ------------------
guide_width = 8   # cm 
guide_height = 8  # cm

radius = 12  # elbow radius
in_offset = radius - guide_width / 2
out_offset = radius - guide_width / 2

# Geometry
geometry = [
    mp.Block(center=mp.Vector3(x=in_offset, y=-Size_y/4),
             size=mp.Vector3(guide_width, Size_y/2, mp.inf),
             material=air),

    mp.Block(center=mp.Vector3(y=out_offset, x=-Size_x/4),
             size=mp.Vector3(Size_x/2, guide_width, mp.inf),
             material=air),

    mp.Wedge(radius=radius, height=mp.inf, material=air,
             wedge_angle=np.pi/2, wedge_start=mp.Vector3(1.0, 0.0, 0.0)),

    mp.Wedge(radius=radius - guide_width, height=mp.inf, material=metal,
             wedge_angle=np.pi/2, wedge_start=mp.Vector3(1.0, 0.0, 0.0))
]


# ------------------ Source ------------------
f_center_hz = 2.45e9  # 2.45 GHz
f_center = f_center_hz / 3e10  # coverts Hz → 1/cm
fwidth = 0.2 * f_center

pulse = mp.GaussianSource(frequency=f_center, fwidth=fwidth)
source_pos_y = -Size_y/2 + pml_size + 0.5

sources = [mp.Source(src=pulse,
                     component=mp.Ez,
                     center=mp.Vector3(x=in_offset, y=source_pos_y),
                     size=mp.Vector3(x=guide_width, z=Size_z),
                     )]


# ------------------ Simulation ------------------
sim = mp.Simulation(cell_size=cell_size,
                    resolution=resolution,
                    boundary_layers=pml_layers,
                    sources=sources,
                    geometry=geometry,
                    default_material=metal)


# ------------------ Geometry View ------------------
sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Size_x, Size_y)))
plt.savefig("waveguide_geom.png")
plt.close()


# ------------------ Animation ------------------
animate = mp.Animate2D(
    fields=mp.Ez,
    normalize=True,
    field_parameters={'alpha': 0.8, 'cmap': 'RdBu', 'interpolation': 'none'},
    boundary_parameters={'hatch': 'o', 'linewidth': 1.5, 'facecolor': 'y', 'edgecolor': 'b', 'alpha': 0.3},
    output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Size_x, Size_y))
)

sim.run(mp.at_every(0.2, animate),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, Size_y/2 - pml_size - 1), 1e-3))

animate.to_mp4(fps=30, filename='metallic_waveguide.mp4')
plt.close()
