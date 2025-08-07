import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Size_x = 30  # cm
Size_y = 30  # cm
cell_size = mp.Vector3(Size_x, Size_y, 0)

resolution = 5  # células por cm (baixa resolução suficiente p/ 2.45 GHz)
pml_layers = []  # paredes refletem — cavidade fechada


# --- Meios ---
air = mp.Medium(epsilon=1.0)
metal = mp.Medium(D_conductivity=1e7)  # condutividade maior


# --- Paredes metálicas da cavidade ---
wall_thickness = 0.5  # cm
geometry = [
    mp.Block(center=mp.Vector3(y=Size_y/2 - wall_thickness/2),
             size=mp.Vector3(Size_x, wall_thickness, mp.inf),
             material=metal),
    mp.Block(center=mp.Vector3(y=-Size_y/2 + wall_thickness/2),
             size=mp.Vector3(Size_x, wall_thickness, mp.inf),
             material=metal),
    mp.Block(center=mp.Vector3(x=Size_x/2 - wall_thickness/2),
             size=mp.Vector3(wall_thickness, Size_y, mp.inf),
             material=metal),
    mp.Block(center=mp.Vector3(x=-Size_x/2 + wall_thickness/2),
             size=mp.Vector3(wall_thickness, Size_y, mp.inf),
             material=metal),
]


# --- Fonte contínua: 2.45 GHz ---
freq_center = 2.45e9 * (1/3e10)  # Convertendo Hz para 1/cm (c = 3e10 cm/s)

sources = [mp.Source(mp.ContinuousSource(frequency=freq_center),
                     component=mp.Ez,
                     center=mp.Vector3(),
                     size=mp.Vector3(0, 0))]  # ponto


# --- Simulação ---
sim = mp.Simulation(cell_size=cell_size,
                    resolution=resolution,
                    geometry=geometry,
                    sources=sources,
                    boundary_layers=pml_layers,
                    default_material=air)


# --- Animação ---
animate = mp.Animate2D(fields=mp.Ez,
                        normalize=True,
                        field_parameters={'alpha': 0.9, 'cmap': 'RdBu'},
                        output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Size_x, Size_y)))

ez_time = []

def get_ez(t):
    pt = mp.Vector3()
    ez_time.append(sim.get_field_point(mp.Ez, pt))

sim.run(mp.at_every(1, animate), mp.at_every(0.2, get_ez), until=1000)

animate.to_mp4(fps=30, filename="resonant_cavity_2-45GHz.mp4")
plt.close()



# --- FFT ---
from scipy.fftpack import fft, fftfreq

dt = 0.2
n = len(ez_time)
t = np.linspace(0, n*dt, n)
ez_fft = fft(ez_time)
freqs = fftfreq(n, dt)

pos_freqs = freqs[:n//2] * 3e10  # converter de 1/cm para Hz
amp = np.abs(ez_fft[:n//2])

plt.figure(figsize=(8,4))
plt.plot(pos_freqs * 1e-9, amp)  # GHz
plt.xlabel("Frequência (GHz)")
plt.ylabel("Amplitude")
plt.title("Espectro de Ressonância da Cavidade")
plt.grid(True)
plt.savefig("resonant_cavity_2-45GHz.png")
plt.close()
