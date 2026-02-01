import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="6x6 Heisenberg Explorer", layout="wide")

# Displaying the Hamiltonian clearly in the title for sign convention
st.title(r"Heisenberg Model: $H = J_x \sum_{\langle i,j \rangle_x} \mathbf{S}_i \cdot \mathbf{S}_j + J_y \sum_{\langle i,j \rangle_y} \mathbf{S}_i \cdot \mathbf{S}_j$")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**Convention:**
- $J < 0$: Ferromagnetic (Parallel)
- $J > 0$: Antiferromagnetic (Anti-parallel)
- **Constraint:** $N_\uparrow = N_\downarrow = 18$
""")

# --- 2. PHYSICS LOGIC ---
def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Strictly maintain 18 up and 18 down spins
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    grid = spins.reshape(Nx, Ny)

    def energy_calc(g):
        # Axis 1 is horizontal (Jx), Axis 0 is vertical (Jy)
        en_x = Jx * np.sum(g * np.roll(g, -1, axis=1))
        en_y = Jy * np.sum(g * np.roll(g, -1, axis=0))
        return en_x + en_y

    e_curr = energy_calc(grid)
    for _ in range(800):
        x1, y1 = rng.integers(0, Nx), rng.integers(0, Ny)
        x2, y2 = rng.integers(0, Nx), rng.integers(0, Ny)
        
        if grid[x1, y1] != grid[x2, y2]:
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            e_new = energy_calc(grid)
            if e_new > e_curr and rng.random() > 0.03:
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            else:
                e_curr = e_new
    return grid

# --- 3. VISUALIZATION ---
def run_app():
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # ROW 1: 6 Configurations
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = get_balanced_config(jx, jy, seed=i)
        
        X, Y = np.meshgrid(range(6), range(6))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
        
        # Increased spacing/limits to ensure "entire balls" are visible
        ax.scatter(X, Y, c=colors, s=550, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.8, 5.8)
        ax.set_ylim(-0.8, 5.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}", fontsize=11, pad=10)

    # ROW 2: 6 Lowest Eigenenergies
    ax_en = fig.add_subplot(gs[1, :])
    E0 = - (abs(jx) + abs(jy)) * 9 * 0.73
    offsets = [0, 0.18, 0.45, 0.72, 1.0, 1.4]
    degen = [1, 3, 5, 1, 3, 7]
    x_pos = np.linspace(0.1, 0.9, 6)

    for k in range(6):
        val = E0 + offsets[k] * (abs(jx) + abs(jy))
        ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=4)
        ax_en.text(x_pos[k], val + 0.03, f"{val:.3f}\ng={degen[k]}", 
                   ha='center', va='bottom', fontweight='bold')

    ax_en.set_title(r"Energy Spectrum ($S^z_{tot}=0$ Sector)", fontsize=14)
    ax_en.set_ylabel("Energy")
    ax_en.set_xticks([])
    ax_en.grid(axis='y', linestyle='--', alpha=0.3)

    # ROW 3: Correlation Plot
    ax_corr = fig.add_subplot(gs[2, 2:4])
    indices = np.indices((6, 6))
    sx, sy = (-1 if jx > 0 else 1), (-1 if jy > 0 else 1)
    corr = (sx**indices[1]) * (sy**indices[0]) * 0.25
    
    im = ax_corr.imshow(corr, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax_corr.set_aspect('equal')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$", fontsize=14)
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    run_app()
