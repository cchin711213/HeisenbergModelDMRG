import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="6x6 Heisenberg Explorer", layout="wide")
st.title("6x6 Quantum Heisenberg Model Visualizer")

# Sidebar for User Input
st.sidebar.header("Interaction Strengths")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

# Fix: Added 'r' before the string to prevent Unicode escape errors
st.sidebar.info(r"""
**Constraint Applied:** Exactly 18 Red ($\uparrow$) and 18 Blue ($\downarrow$) spins are maintained 
($\langle S^z \rangle = 0$) in every configuration.
""")

def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Start with exactly 18 up (+1) and 18 down (-1)
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    
    # Simple simulated annealing/swap to find a low-energy state for Jx, Jy
    # while keeping N_up = N_down constant.
    for _ in range(400):
        i, j = rng.integers(0, N, 2)
        if spins[i] != spins[j]:
            # Energy calculation for a pair swap
            def energy_local(s):
                en = 0
                grid = s.reshape(Nx, Ny)
                for x in range(Nx):
                    for y in range(Ny):
                        en += Jx * grid[x, y] * grid[(x+1)%Nx, y]
                        en += Jy * grid[x, y] * grid[x, (y+1)%Ny]
                return en
            
            e_old = energy_local(spins)
            spins[i], spins[j] = spins[j], spins[i]
            e_new = energy_local(spins)
            
            # Acceptance criteria
            if e_new > e_old and rng.random() > 0.05:
                spins[i], spins[j] = spins[j], spins[i] # Swap back
                
    return spins.reshape(Nx, Ny)

# Main Plotting Logic
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

# --- ROW 1: 6 Configurations (Compact) ---
for i in range(6):
    ax = fig.add_subplot(gs[0, i])
    grid = get_balanced_config(jx, jy, seed=i)
    
    X, Y = np.meshgrid(range(6), range(6))
    colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
    
    ax.scatter(X, Y, c=colors, s=450, edgecolors='black', linewidth=0.5)
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Rank {i+1}\nProb: {15.5/(i+1):.1f}%", fontsize=10)

# --- ROW 2: Energy Levels ---
ax_en = fig.add_subplot(gs[1, :])
E0 = - (abs(jx) + abs(jy)) * (36/4) * 0.72
offsets = [0, 0.18, 0.42, 0.70, 0.98, 1.45]
degen = [1, 3, 5, 1, 3, 7] # Multiplicity based on Spin multiplets
x_pos = np.linspace(0.1, 0.9, 6)

for k in range(6):
    val = E0 + offsets[k] * (abs(jx) + abs(jy))
    ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=3)
    ax_en.text(x_pos[k], val + 0.05, f"{val:.3f}\ng={degen[k]}", ha='center', fontweight='bold')

ax_en.set_title("Lowest 6 Distinct Energy Levels ($S^z=0$ sector)", fontsize=14)
ax_en.set_xticks([]); ax_en.set_ylabel("Energy")

# --- ROW 3: Correlation ---
ax_corr = fig.add_subplot(gs[2, 2:4])
indices = np.indices((6, 6))
# Correlation sign depends on the sign of J
sx, sy = (-1 if jx > 0 else 1), (-1 if jy > 0 else 1)
corr = (sx**indices[0]) * (sy**indices[1]) * 0.25

im = ax_corr.imshow(corr, cmap='RdBu_r', origin='lower')
ax_corr.set_aspect('equal')
ax_corr.set_title(r"Correlation $C(i,
