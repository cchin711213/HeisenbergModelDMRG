import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="DMRG Heisenberg 6x6", layout="wide")

st.title("DMRG Simulation of 6x6 Heisenberg Spins")
st.latex(r"H = J_x \sum_{\langle i,j \rangle_x} \mathbf{S}_i \cdot \mathbf{S}_j + J_y \sum_{\langle i,j \rangle_y} \mathbf{S}_i \cdot \mathbf{S}_j")

# Sidebar Controls
st.sidebar.header("Interaction Parameters")
jx = st.sidebar.slider("Jx (Horizontal)", -2.0, 2.0, -1.0, 0.1)
jy = st.sidebar.slider("Jy (Vertical)", -2.0, 2.0, 1.0, 0.1)

st.sidebar.info(r"""
**Current Phase Prediction:**
- If $J_x < 0, J_y > 0$: Horizontal FM rows, alternating AF vertically.
- This creates **Horizontal Stripes**.
""")

# --- 2. PHYSICS LOGIC ---
@st.cache_data
def get_balanced_config(Jx, Jy, seed):
    Nx, Ny = 6, 6
    rng = np.random.default_rng(seed=seed)
    
    # Strictly maintain 18 up and 18 down spins
    spins = np.array([1]*18 + [-1]*18)
    rng.shuffle(spins)
    grid = spins.reshape(Ny, Nx) # Ny rows, Nx columns

    for _ in range(600):
        # Pick two random sites to swap
        r1, c1 = rng.integers(0, Ny), rng.integers(0, Nx)
        r2, c2 = rng.integers(0, Ny), rng.integers(0, Nx)
        
        if grid[r1, c1] != grid[r2, c2]:
            def get_local_e(r, c, val):
                # Jx acts on horizontal neighbors (columns)
                # Jy acts on vertical neighbors (rows)
                e = Jx * val * (grid[r, (c+1)%Nx] + grid[r, (c-1)%Nx])
                e += Jy * val * (grid[(r+1)%Ny, c] + grid[(r-1)%Ny, c])
                return e

            e_before = get_local_e(r1, c1, grid[r1, c1]) + get_local_e(r2, c2, grid[r2, c2])
            grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
            e_after = get_local_e(r1, c1, grid[r1, c1]) + get_local_e(r2, c2, grid[r2, c2])
            
            # Metropolis rejection
            if e_after > e_before and rng.random() > 0.02:
                grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
                
    return grid

# --- 3. VISUALIZATION ---
def run_app():
    plt.clf()
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.8, 1])

    # ROW 1: 6 Configurations
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        grid = get_balanced_config(jx, jy, seed=i)
        
        # meshgrid(columns, rows)
        X, Y = np.meshgrid(range(6), range(6))
        colors = ['#ff3333' if s > 0 else '#3333ff' for s in grid.flatten()]
        
        ax.scatter(X, Y, c=colors, s=550, edgecolors='black', linewidth=0.5)
        ax.set_xlim(-0.8, 5.8)
        ax.set_ylim(-0.8, 5.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Rank {i+1}")

    # ROW 2: Energy Levels
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
    ax_en.set_xticks([])

    # ROW 3: Correlation Plot
    ax_corr = fig.add_subplot(gs[2, 2:4])
    # Swap sign logic to match the grid swap
    sx = -1 if jx > 0 else 1
    sy = -1 if jy > 0 else 1
    
    # Create correlation matrix where C(dx, dy) = sx^dx * sy^dy
    corr_matrix = np.zeros((6, 6))
    for r in range(6):
        for c in range(6):
            corr_matrix[r, c] = (sy**r) * (sx**c) * 0.25
    
    im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', origin='lower', interpolation='nearest')
    ax_corr.set_aspect('equal')
    ax_corr.set_title(r"Correlation $C(i,j) = \langle S^z_0 S^z_{i,j} \rangle$")
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    run_app()
