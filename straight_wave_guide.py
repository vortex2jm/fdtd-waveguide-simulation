import meep as mp
import matplotlib.pyplot as plt


# -------------------- Parâmetros da simulação --------------------
Size_x = 3
Size_y = 10
Size_z = 0
cell_size = mp.Vector3(Size_x, Size_y, Size_z)

resolution = 25  # cells per micron
pml_size = 10 / resolution
pml_layers = [mp.PML(thickness=pml_size)]

n_core = 3.5  # Silicon
n_clad = 1.44  # SiO2
core = mp.Medium(index=n_core)
cladding = mp.Medium(index=n_clad)

w = 0.5  # largura do guia
h = 0.22  # altura do guia

geometry = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(w, mp.inf, h), material=core),
]


# -------------------- Source --------------------
wl0 = 1.55
min_wl = 1.4
max_wl = 1.7

freq = 1. / wl0
min_freq = 1 / max_wl
max_freq = 1 / min_wl
f_width = max_freq - min_freq
pulse = mp.GaussianSource(freq, fwidth=f_width * 2)

source_pos_y = -Size_y // 2 + pml_size + 0.1

sources = [mp.EigenModeSource(pulse,
                              center=mp.Vector3(y=source_pos_y),
                              size=mp.Vector3(x=Size_x, z=Size_z),
                              direction=mp.Y,
                              eig_band=1)]


# -------------------- Inicializa a simulação --------------------
sim = mp.Simulation(cell_size=cell_size,
                    resolution=resolution,
                    boundary_layers=pml_layers,
                    sources=sources,
                    geometry=geometry)

# -------------------- Salvar cortes 2D estáticos --------------------
sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Size_x, Size_y)))
plt.savefig("simXY.png")
plt.close()


# -------------------- Animação: Captura de frames do Ex --------------------
animate = mp.Animate2D(
  fields=mp.Ez,
  normalize=True,
  field_parameters={'alpha': 0.8, 'cmap':'RdBu', 'interpolation':'none'},
  boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3},
  output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Size_x, Size_y))
)

sim.run(mp.at_every(0.2, animate), until_after_sources=mp.stop_when_fields_decayed(5, mp.Ez, mp.Vector3(0, Size_y/2 - pml_size - 0.1), 1e-4))
animate.to_mp4(fps=30, filename='straight_wave_guide.mp4')
plt.close()
