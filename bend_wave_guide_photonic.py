import meep as mp
import numpy as np
import matplotlib.pyplot as plt


# -------------------- Parameters --------------------
Size_x = 15   # grid size (microns)
Size_y = 15
Size_z = 0
cell_size = mp.Vector3(Size_x,Size_y,Size_z)

resolution = 25 # cells per micron
pml_size = 10 / resolution  # 10 FDTD cells for PML
pml_layers = [mp.PML(thickness=pml_size)]

# Waveguide materials
n_core = 2.5 # Silicon
n_clad = 1.44 # SiO2
core = mp.Medium(index=n_core)
cladding = mp.Medium(index=n_clad)

radius = 5
w = 0.5   # waveguide width
h = 0.22  # waveguide height
in_offset = radius - w/2
out_offset = radius - w/2

# Waveguide bend geometry
geometry = [
  mp.Block(center=mp.Vector3(x = in_offset, y = -Size_y/4), size=mp.Vector3(w, Size_y/2, h), material=core),
  mp.Block(center=mp.Vector3(y = out_offset, x = -Size_x/4), size=mp.Vector3(Size_x/2, w, h), material=core),
  mp.Wedge(radius = radius, height = h, material = core, wedge_angle = np.pi/2,wedge_start=mp.Vector3(1.0, 0.0, 0.0),),
  mp.Wedge(radius = radius - w, height = h, material = cladding, wedge_angle = np.pi/2,wedge_start=mp.Vector3(1.0, 0.0, 0.0))
]


# -------------------- Source --------------------
wl0 = 1.55
min_wl = 1.4
max_wl = 1.7

freq = 1./wl0
min_freq = 1 / max_wl
max_freq = 1 / min_wl
f_width = max_freq - min_freq
pulse = mp.GaussianSource(freq,fwidth=f_width * 2) # gaussian pulse

source_pos_y = - Size_y / 2 + pml_size + 0.1

# Mode source
sources = [mp.EigenModeSource(pulse,
                              center=mp.Vector3(x = in_offset,y = source_pos_y),
                              size=mp.Vector3(x = 4, z = Size_z),
                              direction=mp.Y,
                              eig_band=1,     # first mode
                              )]


# -------------------- Simulation Init --------------------
sim = mp.Simulation(cell_size=cell_size,
                    resolution=resolution,
                    boundary_layers=pml_layers,
                    sources=sources,
                    geometry=geometry,
                    default_material = cladding)


# -------------------- Animation --------------------
animate = mp.Animate2D(
  fields=mp.Ez,
  normalize=True,
  field_parameters={'alpha': 0.9, 'cmap':'viridis', 'interpolation':'none'},
  boundary_parameters={'hatch':'/', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.5},
  output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Size_x, Size_y))
)

sim.run(mp.at_every(0.2, animate), until_after_sources=mp.stop_when_fields_decayed(5, mp.Ez, mp.Vector3(0, Size_y/2 - pml_size - 0.1), 1e-4))
animate.to_mp4(fps=30, filename='bend_wave_guide_photonics_Ez.mp4')
plt.close()
