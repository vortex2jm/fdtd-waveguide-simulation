import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros físicos em centímetros ---
# Dimensões internas da cavidade do micro-ondas
cavity_size_x = 30
cavity_size_y = 30

# Dimensões do guia de onda que alimenta a cavidade
waveguide_width = 8.0
waveguide_length = 15.0

# O tamanho total da célula de simulação precisa incluir a cavidade e o guia de onda
cell_x = cavity_size_x
cell_y = cavity_size_y + waveguide_length
cell_size = mp.Vector3(cell_x, cell_y, 0)

# --- Resolução e Camadas de Contorno ---
resolution = 5  # células por cm (suficiente para 2.45 GHz)

# Adiciona uma camada absorvente (PML) APENAS na extremidade do guia de onda (embaixo)
# As outras bordas (padrão) serão metálicas (refletoras), formando a cavidade.
pml_layers = [mp.PML(thickness=1.0, direction=mp.Y, side=mp.Low)]

# --- Meios ---
air = mp.Medium(epsilon=1.0)
# A condutividade D é usada para metais em meep, simulando perdas por corrente de Foucault
metal = mp.Medium(D_conductivity=1e7) 

# --- Geometria: Cavidade + Guia de Onda ---
wall_thickness = 1.0 # Espessura das paredes metálicas

# Calcula a posição y onde a cavidade começa (acima do guia de onda)
cavity_offset_y = waveguide_length / 2

# Lista de objetos que formam as paredes
geometry = [
    # Paredes da Cavidade (acima do centro da célula)
    # Parede superior
    mp.Block(center=mp.Vector3(0, cavity_offset_y + cavity_size_y/2 - wall_thickness/2),
             size=mp.Vector3(cavity_size_x, wall_thickness, mp.inf),
             material=metal),
    # Parede esquerda
    mp.Block(center=mp.Vector3(-cavity_size_x/2 + wall_thickness/2, cavity_offset_y),
             size=mp.Vector3(wall_thickness, cavity_size_y, mp.inf),
             material=metal),
    # Parede direita
    mp.Block(center=mp.Vector3(cavity_size_x/2 - wall_thickness/2, cavity_offset_y),
             size=mp.Vector3(wall_thickness, cavity_size_y, mp.inf),
             material=metal),

    # Paredes do Guia de Onda (abaixo e conectado à cavidade)
    # Parede inferior da cavidade / superior do guia - com uma abertura
    # Segmento esquerdo
    mp.Block(center=mp.Vector3(-(cavity_size_x + waveguide_width)/(4) + wall_thickness/2, cavity_offset_y - cavity_size_y/2 + wall_thickness/2),
             size=mp.Vector3(cavity_size_x/2 - waveguide_width/2, wall_thickness, mp.inf),
             material=metal),
    # Segmento direito
    mp.Block(center=mp.Vector3((cavity_size_x + waveguide_width)/(4) - wall_thickness/2, cavity_offset_y - cavity_size_y/2 + wall_thickness/2),
             size=mp.Vector3(cavity_size_x/2 - waveguide_width/2, wall_thickness, mp.inf),
             material=metal),

    # Paredes laterais do guia de onda
    mp.Block(center=mp.Vector3(-waveguide_width/2 - wall_thickness/2, -cavity_size_y/2),
             size=mp.Vector3(wall_thickness, waveguide_length, mp.inf),
             material=metal),
    mp.Block(center=mp.Vector3(waveguide_width/2 + wall_thickness/2, -cavity_size_y/2),
             size=mp.Vector3(wall_thickness, waveguide_length, mp.inf),
             material=metal),
]

# --- Fonte de Excitação (dentro do guia de onda) ---
# Frequência de 2.45 GHz (padrão de micro-ondas)
# Convertendo Hz para unidades do meep (1/cm), onde c = 3e10 cm/s
freq_center = 2.45e9 * (1/3e10)

# Posição da fonte: no guia de onda, perto da camada PML
source_pos_y = -cell_y/2 + 2.0 # 2 cm acima da borda inferior

sources = [mp.Source(mp.ContinuousSource(frequency=freq_center),
                      component=mp.Ez,
                      # Posição central da fonte
                      center=mp.Vector3(0, source_pos_y),
                      # Tamanho da fonte: uma linha que cruza o guia para lançar uma onda plana
                      size=mp.Vector3(waveguide_width, 0))]

# --- Simulação ---
sim = mp.Simulation(cell_size=cell_size,
                      resolution=resolution,
                      geometry=geometry,
                      sources=sources,
                      boundary_layers=pml_layers,
                      default_material=air)

# --- Animação ---
# Visualiza a simulação inteira
animate = mp.Animate2D(fields=mp.Ez,
                       normalize=True,
                       field_parameters={'alpha': 0.9, 'cmap':'viridis', 'interpolation':'none'},
                       boundary_parameters={'hatch':'/', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.5},
                       output_plane=mp.Volume(center=mp.Vector3(0, cavity_offset_y - cavity_size_y/2), size=mp.Vector3(cell_x, cell_y)))

ez_time = []
# Ponto para medir o campo para o FFT (centro da cavidade)
fft_point = mp.Vector3(0, cavity_offset_y)

def get_ez(sim):
    ez_time.append(sim.get_field_point(mp.Ez, fft_point))

# Roda a simulação por mais tempo para a onda preencher a cavidade e ressoar
sim.run(mp.at_every(2, animate), mp.at_every(1, get_ez), until=1500)

animate.to_mp4(fps=30, filename="microwave_oven.mp4")
plt.close()


# --- FFT para análise de ressonância ---
from scipy.fftpack import fft, fftfreq

dt = 1 # O passo de tempo da coleta de dados (at_every=1)
n = len(ez_time)
t = np.linspace(0, n*dt, n)
ez_fft = fft(ez_time)

# Converte as frequências do meep para Hz
freqs = fftfreq(n, dt) * (3e10) 

# Pega apenas as frequências positivas e suas amplitudes
pos_freqs = freqs[:n//2]
amp = np.abs(ez_fft[:n//2])

plt.figure(figsize=(10,5))
plt.plot(pos_freqs * 1e-9, amp) # Plotar em GHz
plt.xlabel("Frequência (GHz)")
plt.ylabel("Amplitude do Campo Elétrico (un. arb.)")
plt.title("Espectro de Frequência no Centro da Cavidade")
# Adiciona uma linha vertical na frequência da fonte
plt.axvline(x=2.45, color='r', linestyle='--', label='Frequência da Fonte (2.45 GHz)')
plt.xlim(0, 5) # Limita a visualização para a faixa de interesse
plt.grid(True)
plt.legend()
plt.savefig("microwave_oven_fft.png")
plt.show()
plt.close()
