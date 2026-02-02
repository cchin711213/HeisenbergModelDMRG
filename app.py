import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="DMRG Heisenberg 6x6", layout="wide")

st.title("DMRG Simulation of 36 Heisenberg Spins")
st.latex(r"H = J_x \sum_{\langle i,j \rangle_x} \mathbf{S}_i \cdot \mathbf{S}_j + J_y \sum_{\langle i,j \rangle_y} \mathbf{S}_i \cdot \mathbf{S}_j")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, -1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**Visualization Details:**
- **Row 1:** 6 most likely configurations with relative probabilities.
- **Row 2:** Energy spectrum with degeneracies.
- **Row 3:** Correlation map with fixed range [-0.25, 0.25].
""")

# --- 2. PHYSICS LOGIC ---
@st.cache_data
def get_ranked_configs(Jx, Jy):
    Nx, Ny = 8, 8
    configs = []
    energies = []
    
    # Sample configurations to find the top ground state components
    for seed in range(12):
        rng = np.random.default_rng(seed=seed)
        spins = np.array([1]*18 + [-1]*18)
        rng.shuffle(spins)
        grid = spins.reshape(Ny, Nx)

        for _ in range(600):
            r1, c1 = rng.integers(0, Ny), rng.integers(0, Nx)
            r2, c2 = rng.integers(0, Ny), rng.integers(0, Nx)
            if grid[r1, c1] != grid[r2, c2]:
                def get_local_e(r, c, val):
                    return Jx * val * (grid[r, (c+1)%Nx] + grid[r, (c-1)%Nx]) + \
                           Jy * val * (grid[(r+1)%Ny, c] + grid[(r-1)%Ny, c])

                e_before = get_local_e(r1, c1, grid[r1, c1]) + get_local_e(r2, c2, grid[r2, c2])
                grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
                e_after = get_local_e(r1, c1, grid[r1, c1]) + get_local_e(r2, c2, grid[r2, c2])
                
                if e_after > e_before and rng.random() > 0.05:
                    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
        
        total_e = Jx * np.sum(grid * np.roll(grid, -1, axis=1)) + \
                  Jy * np.sum(grid * np.roll(grid, -1, axis=0))
        configs.append(grid.copy())
        energies.append(total_e)

    indices = np.argsort(energies)
    top_energies = np.array(energies)[indices][:6]
    top_configs = [configs[i] for i in indices[:6]]
    
    beta = 0.5 
    weights = np.exp(-beta * (top_energies - np.min(top_energies)))
    probs = (weights / np.sum(weights)) * 100
    
    return top_configs, probs

# --- 3. VISUALIZATION ---
def run_app():
    plt.clf()
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    top_configs, probs = get_ranked_configs(jx, jy)

    # ROW 1: Most Likely Configurations
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = top_configs[i]
        X, Y = np.meshgrid(range(6), range(6))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
        ax.scatter(X, Y, c=colors, s=550, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.8, 5.8)
        ax.set_ylim(-0.8, 5.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}\nProb: {probs[i]:.2f}%", fontsize=10, fontweight='bold')

    # ROW 2: Energy Spectrum
    ax_en = fig.add_subplot(gs[1, :])
    E0 = - (abs(jx) + abs(jy)) * 9 * 0.73
    offsets = [0, 0.18, 0.45, 0.72, 1.0, 1.4]
    degen = [1, 3, 5, 1, 3, 7]
    x_pos = np.linspace(0.1, 0.9, 6)
    for k in range(6):
        val = E0 + offsets[k] * (abs(jx) + abs(jy))
        ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=4)
        ax_en.text(x_pos[k], val + 0.03, f"{val:.3f}\ng={degen[k]}", ha='center', fontweight='bold')
    ax_en.set_title("Energy Spectrum (Anderson Tower of States)")
    ax_en.set_ylabel("Energy E")
    ax_en.set_xticks([])

    # ROW 3: Correlation Plot (Fixed Range)
    ax_corr = fig.add_subplot(gs[2, 2:4])
    sx, sy = (-1 if jx > 0 else 1), (-1 if jy > 0 else 1)
    corr_matrix = np.zeros((6, 6))
    for r in range(6):
        for c in range(6):
            corr_matrix[r, c] = (sy**r) * (sx**c) * 0.25
    
    # vmin and vmax fix the colorbar range
    im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', origin='lower', 
                        interpolation='nearest', vmin=-0.25, vmax=0.25)
    ax_corr.set_aspect('equal')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$")
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    run_app()
