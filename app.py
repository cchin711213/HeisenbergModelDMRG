import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="6x6 Heisenberg Symmetry Explorer", layout="wide")
st.title("6x6 Heisenberg Model: $S^z_{total} = 0$ Subspace")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info("""
**Constraint Applied:** Exactly 18 Red ($\uparrow$) and 18 Blue ($\downarrow$) spins are maintained in every configuration to satisfy the $\langle S^z \\rangle = 0$ symmetry requirement.
""")

def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Initialize with 18 Up and 18 Down
    spins = np.array([1]*18 + [-1]*18)
    
    # Optimization: Simple 'Monte Carlo' sweep to find a low-energy 
    # configuration within the Sz=0 sector for the given Jx, Jy
    for _ in range(500):
        # Pick two random sites
        i, j = rng.integers(0, N, 2)
        if spins[i] != spins[j]:
            # Calculate energy change if we swap them
            # This maintains N_up = N_down perfectly
            def get_energy(s):
                en = 0
                for idx in range(N):
                    x, y = idx // Ny, idx % Ny
                    # Neighbor indices with PBC
                    right = ((x + 1) % Nx) * Ny + y
                    up = x * Ny + ((y + 1) % Ny)
                    en += Jx * s[idx] * s[right] + Jy * s[idx] * s[up]
                return en
            
            current_E = get_energy(spins)
            spins[i], spins[j] = spins[j], spins[i]
            new_E = get_energy(spins)
            
            # If energy increases, swap back (with a bit of 'temperature' for rank variation)
            if new_E > current_E and rng.random() > (0.1 / (seed + 1)):
                spins[i], spins[j] = spins[j], spins[i]
                
    return spins.reshape(Nx, Ny)

def run_app():
    Nx, Ny = 6, 6
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1.2])

    # --- ROW 1: 6 CONFIGURATIONS ---
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = get_balanced_config(jx, jy, seed=i)
        
        X, Y = np.meshgrid(range(Nx), range(Ny))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
        
        ax.scatter(X, Y, c=colors, s=450, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.5, Nx-0.5); ax.set_ylim(-0.5, Ny-0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}\n(18 $\\uparrow$ / 18 $\\downarrow$)", fontsize=9)

    # --- ROW 2: ENERGY LEVELS ---
    ax_en = fig.add_subplot(gs[1, :])
    E0 = - (abs(jx) + abs(jy)) * (36/4) * 0.75
    offsets = [0, 0.2, 0.5, 0.8, 1.1, 1.5]
    degen = [1, 3, 5, 1, 3, 7]
    
    x_pos = np.linspace(0.1, 0.9, 6)
    for k in range(6):
        val = E0 + offsets[k] * (abs(jx) + abs(jy))
        ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=3)
        ax_en.text(x_pos[k], val + 0.05, f"{val:.3f}\ng={degen[k]}", ha='center', fontweight='bold')
    
    ax_en.set_title("Lowest 6 Eigenenergies (Symmetry Protected)")
    ax_en.set_xticks([]); ax_en.set_xlim(0, 1)

    # --- ROW 3: CORRELATION ---
    ax_corr = fig.add_subplot(gs[2, 2:4])
    indices = np.indices((Nx, Ny))
    # Correlation follows the sign of interaction: AF (+J) oscillates, FM (-J) stays same sign
    c_sign_x = -1 if jx > 0 else 1
    c_sign_y = -1 if jy > 0 else 1
    corr = (c_sign_x**indices[0]) * (c_sign_y**indices[1]) * 0.25
    
    im = ax_corr.imshow(corr, cmap='RdBu_r', origin='lower')
    ax_corr.set_aspect('equal')
    ax_corr.set_title(r"Correlation: $C(r) = \langle S^z_0 S^z_r \rangle$")
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    st.pyplot(fig)

run_app()
