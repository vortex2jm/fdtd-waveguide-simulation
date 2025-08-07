import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep import mpb

mp.verbosity(0)

# -------- Parâmetros do material --------
n_core = 2.5   # índice do núcleo (Silício)
n_clad = 1.44  # índice do revestimento (SiO2)
core = mp.Medium(index=n_core)
cladding = mp.Medium(index=n_clad)

# -------- Simulation geometry --------
resolution = 100  # pixels/μm
geometry_lattice = mp.Lattice(size=mp.Vector3(3, 2))  # tamanho da simulação em μm

w = 0.7   # largura do núcleo (μm)
h = 0.25  # altura do núcleo (μm)

geometry = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, mp.inf), material=cladding),  # fundo de SiO2
    mp.Block(center=mp.Vector3(), size=mp.Vector3(w, h), material=core)  # guia retangular
]

# -------- Parâmetros ópticos --------
wl0 = 1.55         # comprimento de onda em μm
freq = 1.0 / wl0   # frequência correspondente
num_modes = 1      # número de modos a calcular

# -------- Solver config --------
ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_modes,
)

ms.init_params(mp.NO_PARITY, True)

# -------- Permittivity view --------
eps = ms.get_epsilon()
x = np.arange(eps.shape[0]) / resolution
y = np.arange(eps.shape[1]) / resolution

plt.contourf(x, y, eps.transpose(), cmap='binary')
plt.colorbar(label="Permittivity")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.title("Distribuição da Permissividade (ε)")
plt.savefig("Permittivity.png")
plt.close()

# -------- Cálculo dos modos --------
n_mode_guess = 0.5 * (n_core + n_clad)
k_guess = freq * n_mode_guess
k_min = freq * n_clad
k_max = freq * n_core
tol = 1e-5
neff = []
cmap = 'viridis'

for mode_num in range(1, num_modes + 1):
    # Finds the mode
    k_mpb = ms.find_k(mp.NO_PARITY, freq, mode_num, mode_num, mp.Vector3(0, 0, 1), tol, k_guess, k_min, k_max)
    neff_val = k_mpb[0] / freq
    neff.append(neff_val)

    # Fields
    E = ms.get_efield(which_band=mode_num)
    H = ms.get_hfield(which_band=mode_num)
    P = ms.get_poynting(which_band=mode_num)

    Ex = E[:,:,0,0]
    Ey = E[:,:,0,1]
    Ez = E[:,:,0,2]
    Pz = 0.5 * np.real(P[:, :, 0, 2])

    plt.contourf(x, y, np.abs(Ex.transpose()), 202, cmap=cmap)
    plt.colorbar()
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.title(f"Ex do modo {mode_num}")
    plt.savefig(f"Ex_mode_{mode_num}.png")
    plt.close()

    plt.contourf(x, y, np.abs(Ey.transpose()), 202, cmap=cmap)
    plt.colorbar()
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.title(f"Ey do modo {mode_num}")
    plt.savefig(f"Ey_mode_{mode_num}.png")
    plt.close()

    plt.contourf(x, y, np.abs(Ez.transpose()), 202, cmap=cmap)
    plt.colorbar()
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.title(f"Ez do modo {mode_num}")
    plt.savefig(f"Ez_mode_{mode_num}.png")
    plt.close()

    plt.contourf(x, y, Pz.transpose(), 202, cmap=cmap)
    plt.colorbar()
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.title(f"Poynting z do modo {mode_num}")
    plt.savefig(f"Poynting_mode_{mode_num}.png")
    plt.close()

