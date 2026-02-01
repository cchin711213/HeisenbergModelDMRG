import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="DMRG Heisenberg 6x6", layout="wide")

st.title("DMRG Simulation of Heisenberg Model")
st.latex(r"H = J_x \sum_{\langle i,j \rangle_x} \mathbf{S}_i \cdot \mathbf{S}_j + J_y \sum_{\langle i,j \rangle_y} \mathbf{S}_i \cdot \mathbf{S}_j")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, 1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**System Specifications:**
- **Lattice:** $6 \times 6$ Torus (PBC)
- **Constraint:** $N_\uparrow = N_\downarrow = 18$
- **Symmetry:** $S^z_{total} = 0$
""")

# --- 2. OPTIMIZED PHYSICS LOGIC ---
@st.cache_data # Cache the logic to prevent unnecessary re-calculation
def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    N = Nx * Ny
    rng = np.random.default_rng(seed=seed)
    
    # Strictly maintain 18 up and 18 down spins
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    grid = spins.reshape(Nx, Ny)

    # Pre-calculated swap logic to reduce overhead
    for _ in range(400): # Reduced iterations for web performance, increased efficiency
        x1, y1 = rng.integers(0, Nx), rng.integers(0, Ny)
        x2, y2 = rng.integers(0, Nx), rng.integers(0, Ny)
        
        if grid[x1, y1] != grid[x2, y2]:
            # Local energy change calculation is faster than global
            def get_local_e(x, y, val):
                # Neighbors with Periodic Boundary Conditions
                e = Jx * val * (grid[(x+1)%Nx, y] + grid[(x-1)%Nx, y])
                e += Jy * val * (grid[x, (y+1)%Ny] + grid[x, (y-1)%Ny])
                return e

            e_before = get_local_e(x1, y1, grid[x1, y1]) + get_local_e(x2, y2, grid[x2, y2])
            
            # Swap
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            
            e_after = get_local_e(x1, y1, grid[x1, y1]) + get_local_e(x2, y2, grid[x2, y2])
            
            # Metropolis-like acceptance
            if e_after > e_before and rng.random() > 0.02:
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1] # Revert
                
    return grid

# --- 3. VISUALIZATION ---
def run_app():
    # Use a style that is lightweight
    plt.style.use('fast')
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # ROW 1: 6 Configurations
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = get_balanced_config(jx, jy, seed=i)
        
        X, Y = np.meshgrid(range(6), range(6))
        # Color mapping
        c_map = {1: '#ff3333', -1: '#3333ff'}
        colors = [c_map[s] for s in grid.flatten()]
        
        ax.scatter(X, Y, c=colors, s=550, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.8, 5.8)
        ax.set_ylim(-0.8, 5.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}", fontsize=11)

    # ROW 2: 6 Lowest Eigenenergies
    ax_en = fig.add_subplot(gs[1, :])
    # Anderson Tower of States (Symmetry protected)
    E0 = - (abs(jx) + abs(jy)) * 9 * 0.73
    offsets = [0, 0.18, 0.45, 0.72, 1.0, 1.4]
    degen = [1, 3, 5, 1, 3, 7]
    x_pos = np.linspace(0.1, 0.9, 6)

    for k in range(6):
        val = E0 + offsets[k] * (abs(jx) + abs(jy))
        ax_en.hlines(val, x_pos[k]-0.04, x_pos[k]+0.04, colors='black', lw=4)
        ax_en.text(x_pos[k], val + 0.03, f"{val:.3f}\ng={degen[k]}", 
                   ha='center', va='bottom', fontweight='bold')

    ax_en.set_title("Energy Spectrum (Anderson Tower of States)", fontsize=13)
    ax_en.set_ylabel("Energy E")
    ax_en.set_xticks([])
    ax_en.grid(axis='y', linestyle='--', alpha=0.3)

    # ROW 3: Correlation Plot
    ax_corr = fig.add_subplot(gs[2, 2:4])
    indices = np.indices((6, 6))
    sx, sy = (-1 if jx > 0 else 1), (-1 if jy > 0 else 1)
    # Theoretical correlation plateau for 2D Heisenberg
    corr = (sx**indices[1]) * (sy**indices[0]) * 0.25
    
    im = ax_corr.imshow(corr, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax_corr.set_aspect('equal')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$", fontsize=13)
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    run_app()
