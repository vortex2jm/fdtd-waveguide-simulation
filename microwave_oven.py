import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# --- Physic parameters ---
cavity_size_x = 30
cavity_size_y = 30

waveguide_width = 8.0
waveguide_length = 15.0

cell_x = cavity_size_x
cell_y = cavity_size_y + waveguide_length
cell_size = mp.Vector3(cell_x, cell_y, 0)


resolution = 5  # cells per cm
pml_layers = [mp.PML(thickness=1.0, direction=mp.Y, side=mp.Low)]


# -------------- Medium -------------
air = mp.Medium(epsilon=1.0)
metal = mp.Medium(D_conductivity=1e7)   # high conductivity 

wall_thickness = 1.0 
cavity_offset_y = waveguide_length / 2


# -------------- Geometry -------------
geometry = [
    mp.Block(center=mp.Vector3(0, cavity_offset_y + cavity_size_y/2 - wall_thickness/2),
             size=mp.Vector3(cavity_size_x, wall_thickness, mp.inf),
             material=metal),

    mp.Block(center=mp.Vector3(-cavity_size_x/2 + wall_thickness/2, cavity_offset_y),
             size=mp.Vector3(wall_thickness, cavity_size_y, mp.inf),
             material=metal),

    mp.Block(center=mp.Vector3(cavity_size_x/2 - wall_thickness/2, cavity_offset_y),
             size=mp.Vector3(wall_thickness, cavity_size_y, mp.inf),
             material=metal),

    mp.Block(center=mp.Vector3(-(cavity_size_x + waveguide_width)/(4) + wall_thickness/2, cavity_offset_y - cavity_size_y/2 + wall_thickness/2),
             size=mp.Vector3(cavity_size_x/2 - waveguide_width/2, wall_thickness, mp.inf),
             material=metal),

    mp.Block(center=mp.Vector3((cavity_size_x + waveguide_width)/(4) - wall_thickness/2, cavity_offset_y - cavity_size_y/2 + wall_thickness/2),
             size=mp.Vector3(cavity_size_x/2 - waveguide_width/2, wall_thickness, mp.inf),
             material=metal),

    mp.Block(center=mp.Vector3(-waveguide_width/2 - wall_thickness/2, -cavity_size_y/2),
             size=mp.Vector3(wall_thickness, waveguide_length, mp.inf),
             material=metal),

    mp.Block(center=mp.Vector3(waveguide_width/2 + wall_thickness/2, -cavity_size_y/2),
             size=mp.Vector3(wall_thickness, waveguide_length, mp.inf),
             material=metal),
]


# -------------- Source -------------

# microwaves - 2.45GHz
freq_center = 2.45e9 * (1/3e10)     # Hz to meep unit (1/cm)

source_pos_y = -cell_y/2 + 2.0

sources = [mp.Source(mp.ContinuousSource(frequency=freq_center),
                      component=mp.Ez,
                      center=mp.Vector3(0, source_pos_y),
                      size=mp.Vector3(waveguide_width, 0))]


# -------------- Simulation ----------
sim = mp.Simulation(cell_size=cell_size,
                      resolution=resolution,
                      geometry=geometry,
                      sources=sources,
                      boundary_layers=pml_layers,
                      default_material=air)


# -------------- Animation -------------
animate = mp.Animate2D(fields=mp.Ez,
                       normalize=True,
                       field_parameters={'alpha': 0.9, 'cmap':'viridis', 'interpolation':'none'},
                       boundary_parameters={'hatch':'/', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.5},
                       output_plane=mp.Volume(center=mp.Vector3(0, cavity_offset_y - cavity_size_y/2), size=mp.Vector3(cell_x, cell_y)))


# fft data
ez_time = []
fft_point = mp.Vector3(0, cavity_offset_y)

def get_ez(sim):
    ez_time.append(sim.get_field_point(mp.Ez, fft_point))

# 1500 steps simulation
sim.run(mp.at_every(2, animate), mp.at_every(1, get_ez), until=1500)

animate.to_mp4(fps=30, filename="microwave_oven.mp4")
plt.close()


# -------------- FFT -------------
from scipy.fftpack import fft, fftfreq

dt = 1  # time step to data collection
n = len(ez_time)
t = np.linspace(0, n*dt, n)
ez_fft = fft(ez_time)

# Convert meep to Hz
freqs = fftfreq(n, dt) * (3e10) 

pos_freqs = freqs[:n//2]
amp = np.abs(ez_fft[:n//2])

plt.figure(figsize=(10,5))
plt.plot(pos_freqs * 1e-9, amp) # GHz scale
plt.xlabel("Frequência (GHz)")
plt.ylabel("Amplitude do Campo Elétrico (un. arb.)")
plt.title("Espectro de Frequência no Centro da Cavidade")
plt.axvline(x=2.45, color='r', linestyle='--', label='Frequência da Fonte (2.45 GHz)')
plt.xlim(0, 5)
plt.grid(True)
plt.legend()
plt.savefig("microwave_oven_fft.png")
plt.show()
plt.close()
